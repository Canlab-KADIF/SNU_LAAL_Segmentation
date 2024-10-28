# Standardized Max Logits

A simple yet effective approach for identifying unexpected road obstacles in urban-scene segmentation.

## Table of Contents

- [Datasets](#datasets)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Evaluation](#how-to-run-the-evaluation)
  - [Key Arguments](#key-arguments)
- [Visualization](#visualization)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)

## Datasets

- **RoadAnomaly**
- **SML_Fishyscapes_Static**
- **SML_Fishyscapes_LostAndFound**

## Setup and Installation

1. **Navigate to the Standardized Max Logits directory:**

    ```bash
    cd standardized-max-logits
    ```

2. **Install required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download pre-trained models and place them in the `pretrained` directory:**

    - [Download Pretrained Weights](https://drive.google.com/file/d/1Rqty9pRhGdfhkfqlWbFUFgdFp0DvfORN/view)

4. **Download test datasets and organize them as follows:**

    ```
    <dataset_root>
    ├── original/        # Original images
    └── labels/          # Corresponding ground truth masks
    ```

## How to Run the Evaluation

To run the evaluation on a specific OoD dataset, use the following command:

```bash
python eval.py --ood_dataset_path your_dataset_path/SML_Fishyscapes/Static --ood_dataset_name SML_Fishyscapes_Static --score_mode max_logit
```

### Key Arguments

- `--ood_dataset_path`: Path to the directory containing the OoD dataset.
- `--ood_dataset_name`: Name of the OoD dataset (`SML_Fishyscapes_Static`, `SML_Fishyscapes_LostAndFound`, or `road_anomaly`).
- `--score_mode`: Method for calculating the anomaly score (`max_logit` or `standardized_max_logit`).

## Visualization

Visualization outputs are saved in the `segmentation_results` and `anomaly_maps` folders within the specified dataset directory.

### Visualization Includes:

- **Segmentation Results:** Visualizes the predicted class labels.
- **Anomaly Heatmap:** Displays the normalized anomaly scores.
- **Anomaly Binary Map:** Shows a binary map of anomalies based on a specified threshold.

### Directory Structure for Visualization Results:

```
<ood_dataset_path>
├── segmentation_results/
│   └── image_name_max_logit_segmentation.png
├── anomaly_maps/
│   ├── image_name_max_logit_anomaly_map.png
│   └── image_name_max_logit_anomaly_binary.png
```

## Evaluation Metrics

The evaluation includes the following metrics for measuring OoD detection performance:

- **AUROC:** Area Under the Receiver Operating Characteristic curve.
- **AUPRC:** Area Under the Precision-Recall curve.
- **FPR@TPR95:** False Positive Rate at 95% True Positive Rate.

The results will be printed at the end of the evaluation:

```
AUROC score: <value>
AUPRC score: <value>
FPR@TPR95: <value>
```

## Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{Jung_2021_ICCV,
    author    = {Jung, Sanghun and Lee, Jungsoo and Gwak, Daehoon and Choi, Sungha and Choo, Jaegul},
    title     = {Standardized Max Logits: A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
}
```
