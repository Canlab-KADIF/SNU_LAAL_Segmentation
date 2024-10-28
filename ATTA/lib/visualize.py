import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_color_map(palette=None):
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

def save_visualization(image, true_label, pred_label, image_idx, dataset_name, method_name, atta_logit=None, anomaly_score=None, threshold=None):
    # Save paths
    save_path = os.path.join('visualize', dataset_name, method_name)
    label_save_path = os.path.join(save_path, 'label')
    anomaly_score_save_path = os.path.join(save_path, 'anomaly_score')
    atta_anomaly_save_path = os.path.join(save_path, 'atta_anomaly')
    pred_label_save_path = os.path.join(save_path, 'anomaly_class')

    os.makedirs(label_save_path, exist_ok=True)
    os.makedirs(anomaly_score_save_path, exist_ok=True)
    os.makedirs(atta_anomaly_save_path, exist_ok=True)
    os.makedirs(pred_label_save_path, exist_ok=True)

    # Visualize true label
    true_label_mask = (true_label > 0).astype(np.uint8) * 255
    Image.fromarray(true_label_mask).save(os.path.join(label_save_path, f'label_{image_idx+1}.png'))

    # Heatmap visualizing with anomaly score
    if anomaly_score is not None:
        plt.figure()
        plt.imshow(anomaly_score, cmap='bwr')   
        plt.axis('off')
        plt.savefig(os.path.join(anomaly_score_save_path, f'anomaly_score_{image_idx+1}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        print(f"No anomaly score provided for image {image_idx+1}. Skipping anomaly score visualization.")

    # Anomaly detection visualizing with black&white
    if threshold is not None and anomaly_score is not None:
        anomaly_mask = (anomaly_score > threshold).astype(np.uint8) * 255
        Image.fromarray(anomaly_mask).save(os.path.join(atta_anomaly_save_path, f'atta_anomaly_{image_idx+1}.png'))
    else:
        print(f"Anomaly threshold not provided or anomaly score missing for image {image_idx+1}. Skipping anomaly visualization.")
    
    # Color map for visualize class
    if dataset_name in ['FS_Static', 'FS_LostAndFound', 'RoadAnomaly']:
        default_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
                           153, 153, 153, 250, 170, 30,
                           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
                           255, 0, 0, 0, 0, 142, 0, 0, 70,
                           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        palette = create_color_map(default_palette)

        # Predicted label visualization with palette map
        output_image = Image.fromarray(pred_label.astype(np.uint8))
        output_image.putpalette(palette)
        
        # Add light sky blue color on predicted anomaly parts
        if threshold is not None and anomaly_score is not None:
            anomaly_mask = (anomaly_score > threshold)
            colored_pred_label = np.array(output_image.convert("RGB"))  
            colored_pred_label[anomaly_mask] = [135, 206, 250]  
            output_image = Image.fromarray(colored_pred_label.astype(np.uint8)) 
        
        output_image.save(os.path.join(pred_label_save_path, f'anomaly_class_{image_idx+1}.png'))
        print(f"Saved prediction label color map for Fishyscapes in image {image_idx+1}, with anomalies in light sky blue.")