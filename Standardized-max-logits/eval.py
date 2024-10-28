"""
Evaluation Scripts with Segmentation Result Saving
"""
from __future__ import absolute_import
from __future__ import division
from collections import namedtuple, OrderedDict
import argparse
import logging
import os
import torch
import time
import numpy as np

from config import cfg, assert_and_infer_cfg
import network
import optimizer
from ood_metrics import fpr_at_95_tpr
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torchvision.transforms as standard_transforms
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
pretrained_model_path = os.path.join(dirname, 'pretrained/r101_os8_base_cty.pth')

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')

# Additional arguments for your specific setup
parser.add_argument('--syncbn', action='store_true', default=False)
parser.add_argument('--class_uniform_pct', type=float, default=0.0)
parser.add_argument('--batch_weighting', action='store_true', default=False)
parser.add_argument('--jointwtborder', action='store_true', default=False)
parser.add_argument('--strict_bdr_cls', type=str, default='')
parser.add_argument('--rlx_off_iter', type=int, default=-1)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8')
parser.add_argument('--dataset', type=str, default='cityscapes')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--trunk', type=str, default='resnet101')
parser.add_argument('--bs_mult', type=int, default=2)
parser.add_argument('--bs_mult_val', type=int, default=1)
parser.add_argument('--snapshot', type=str, default=pretrained_model_path)
parser.add_argument('--ood_dataset_path', type=str, default='your_dataset_path')
parser.add_argument('--ood_dataset_name', type=str, default='SML_Fishyscapes_Static', help='Name of the OoD dataset')
parser.add_argument('--score_mode', type=str, default='standardized_max_logit')
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--enable_boundary_suppression', type=bool, default=True)
parser.add_argument('--boundary_width', type=int, default=4)
parser.add_argument('--boundary_iteration', type=int, default=4)
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True)
parser.add_argument('--smoothing_kernel_size', type=int, default=7)
parser.add_argument('--smoothing_kernel_dilation', type=int, default=6)

args = parser.parse_args()

# Set random seeds for reproducibility
random_seed = cfg.RANDOM_SEED
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

def get_net():
    """
    Load the network with pretrained weights.
    """
    assert_and_infer_cfg(args)
    net = network.get_net(args, criterion=None, criterion_aux=None)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, None, None,
                            args.snapshot, args.restore_optimizer)
        print(f"Loading completed. Epoch {epoch} and mIoU {mean_iu}")
    else:
        raise ValueError(f"snapshot argument is not set!")

    # Load class statistics
    class_mean = np.load(f'stats/{args.dataset}_mean.npy', allow_pickle=True)
    class_var = np.load(f'stats/{args.dataset}_var.npy', allow_pickle=True)
    net.module.set_statistics(mean=class_mean.item(), var=class_var.item())

    torch.cuda.empty_cache()
    net.eval()

    return net

def preprocess_image(x, mean_std):
    """
    Preprocess the input image.
    """
    x = Image.fromarray(x)
    x = standard_transforms.ToTensor()(x)
    x = standard_transforms.Normalize(*mean_std)(x)
    x = x.cuda()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x

def create_color_map(palette=None):
    """
    Create a color map for visualizing the segmentation results.
    If a palette is provided, use it. Otherwise, generate a random palette.
    """
    if palette is not None:
        zero_pad = 256 * 3 - len(palette)
        palette.extend([0] * zero_pad)
        return palette
    else:
        n_classes = 19  
        palette = []
        for i in range(n_classes):
            palette.extend([(i * 37) % 256, (i * 58) % 256, (i * 159) % 256])
        zero_pad = 256 * 3 - len(palette)
        palette.extend([0] * zero_pad)
        return palette

def save_segmentation_image(output, save_path, palette):
    """
    Save the segmentation result as an image.
    """
    output = output.squeeze(0).cpu().numpy()
    output_image = Image.fromarray(output.astype(np.uint8))
    output_image.putpalette(palette)
    output_image.save(save_path)
    print(f'Segmentation image saved at {save_path}')

def get_image_and_mask_paths(image_root_path, mask_root_path, dataset_name):
    image_paths = []
    mask_paths = []
    if dataset_name == 'road_anomaly':
        image_extension = '.jpg'
        image_files = [f for f in os.listdir(image_root_path) if f.endswith(image_extension)]
        for image_file in image_files:
            image_path = os.path.join(image_root_path, image_file)
            mask_file = image_file.replace('.jpg', '.png')
            mask_path = os.path.join(mask_root_path, mask_file)
            image_paths.append(image_path)
            mask_paths.append(mask_path)
    elif dataset_name in ['SML_Fishyscapes_Static', 'SML_Fishyscapes_LostAndFound']:
        image_extension = '.png'
        image_files = [f for f in os.listdir(image_root_path) if f.endswith(image_extension)]
        for image_file in image_files:
            image_path = os.path.join(image_root_path, image_file)
            mask_file = image_file  
            mask_path = os.path.join(mask_root_path, mask_file)
            image_paths.append(image_path)
            mask_paths.append(mask_path)
    else:
        return [], [], ''
    return image_paths, mask_paths, image_extension

if __name__ == '__main__':
    net = get_net()
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ood_data_root = args.ood_dataset_path
    image_root_path = os.path.join(ood_data_root, 'original')
    mask_root_path = os.path.join(ood_data_root, 'labels')
    dataset_name = args.ood_dataset_name

    if not os.path.exists(image_root_path):
        raise ValueError(f"Dataset directory {image_root_path} doesn't exist!")

    anomaly_score_list = []
    ood_gts_list = []

    image_paths, mask_paths, image_extension = get_image_and_mask_paths(image_root_path, mask_root_path, dataset_name)
    segmentation_result_dir = os.path.join(ood_data_root, 'segmentation_results')
    os.makedirs(segmentation_result_dir, exist_ok=True)
    anomaly_map_dir = os.path.join(ood_data_root, 'anomaly_maps')
    os.makedirs(anomaly_map_dir, exist_ok=True)

    if dataset_name in ['SML_Fishyscapes_Static', 'SML_Fishyscapes_LostAndFound']:
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
                   153, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
                   255, 0, 0, 0, 0, 142, 0, 0, 70,
                   0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        palette = create_color_map(palette)
    else:
        palette = create_color_map()

    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        image_file = os.path.basename(image_path)
        image = np.array(Image.open(image_path).convert('RGB')).astype('uint8')
        mask = Image.open(mask_path)
        ood_gts = np.array(mask)

        ood_gts_list.append(np.expand_dims(ood_gts, 0))

        with torch.no_grad():
            image_tensor = preprocess_image(image, mean_std)
            main_out, anomaly_score = net(image_tensor)

            segmentation_save_path = os.path.join(segmentation_result_dir, image_file.replace(image_extension, 'segmentation.png'))
            save_segmentation_image(main_out.argmax(dim=1), segmentation_save_path, palette)

            anomaly_map = anomaly_score.squeeze().cpu().numpy()
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            anomaly_map_norm = (anomaly_map_norm * 255).astype(np.uint8)
            anomaly_map_image = Image.fromarray(anomaly_map_norm)
            anomaly_map_save_path = os.path.join(anomaly_map_dir, image_file.replace(image_extension, 'anomaly_map.png'))
            anomaly_map_image.save(anomaly_map_save_path)

            threshold = 0.5
            anomaly_binary = (anomaly_map_norm < (threshold * 255)).astype(np.uint8) * 255
            anomaly_binary_image = Image.fromarray(anomaly_binary)
            anomaly_binary_save_path = os.path.join(anomaly_map_dir, image_file.replace(image_extension, 'anomaly_binary.png'))
            anomaly_binary_image.save(anomaly_binary_save_path)

        anomaly_score_list.append(anomaly_score.cpu().numpy())

    ood_gts = np.concatenate(ood_gts_list, axis=0)
    anomaly_scores = np.concatenate(anomaly_score_list, axis=0)

    if dataset_name == 'road_anomaly':
        ood_mask = (ood_gts != 0)
        ind_mask = (ood_gts == 0)
    elif dataset_name in ['SML_Fishyscapes_Static', 'SML_Fishyscapes_LostAndFound']:
        ood_mask = (ood_gts == 1) | (ood_gts == 255)
        ind_mask = (ood_gts == 0)
    else:
        ood_mask = (ood_gts == 1)  # Adjust as necessary for other datasets
        ind_mask = (ood_gts == 0)

    ood_out = -1 * anomaly_scores[ood_mask]
    ind_out = -1 * anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    print('Measuring metrics...')
    fpr = fpr_at_95_tpr(val_label, val_out)
    fpr_full, tpr_full, _ = roc_curve(val_label, val_out)
    roc_auc = auc(fpr_full, tpr_full)
    precision, recall, _ = precision_recall_curve(val_label, val_out)
    prc_auc = average_precision_score(val_label, val_out)

    print(f'AUROC score: {roc_auc}')
    print(f'AUPRC score: {prc_auc}')
    print(f'FPR@TPR95: {fpr}')
