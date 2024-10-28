# SML_ATTA

Combining **Standardized Max Logits** and **ATTA** for robust Out-of-Distribution (OoD) detection in urban-scene segmentation.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Setup and Installation](#setup-and-installation)
- [Standardized Max Logits](#standardized-max-logits)
  - [Setup](#setup)
  - [How to Run the Evaluation](#how-to-run-the-evaluation)
  - [Visualization](#visualization)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Citation](#citation)
- [ATTA](#atta)
  - [Setup and Dataset](#setup-and-dataset)
  - [How to Implement ATTA with ResNet101 Backbone](#how-to-implement-atta-with-resnet101-backbone)
  - [Visualization](#visualization-1)
  - [Explanation](#explanation)
  - [Citation](#citation-1)
- [Combined Usage](#combined-usage)

## Introduction

**SML_ATTA** integrates two powerful methods—**Standardized Max Logits** and **ATTA**—to enhance the detection of unexpected road obstacles in urban-scene segmentation tasks. This unified framework leverages the strengths of both approaches to provide reliable Out-of-Distribution (OoD) detection, ensuring safer and more accurate autonomous driving systems.

## Datasets

Both **Standardized Max Logits** and **ATTA** utilize the following datasets for evaluation:

- **RoadAnomaly**
- **SML_Fishyscapes_Static**
- **SML_Fishyscapes_LostAndFound**

## Setup and Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-repo/SML_ATTA.git
    cd SML_ATTA
    ```

2. **Install Required Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download Pre-trained Models:**

    - **Standardized Max Logits:**
      - [Download Pretrained Weights](https://drive.google.com/file/d/1Rqty9pRhGdfhkfqlWbFUFgdFp0DvfORN/view)
      - Place the downloaded models in the `standardized-max-logits/pretrained` directory.

    - **ATTA:**
      - [Download Pretrained Weights](https://drive.google.com/file/d/1Rqty9pRhGdfhkfqlWbFUFgdFp0DvfORN/view)
      - Place the downloaded weights in the `ATTA/pretrained` directory.

4. **Download Test Datasets:**

    ```bash
    wget http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar
    tar -xvf data_processed.tar
    ```

    Organize the datasets as follows:

    ```
    <dataset_root>
    ├── original/        # Original images
    └── labels/          # Corresponding ground truth masks
    ```

## Standardized Max Logits

A simple yet effective approach for identifying unexpected road obstacles in urban-scene segmentation.

### Setup

1. **Navigate to the Standardized Max Logits Directory:**

    ```bash
    cd standardized-max-logits
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download Pre-trained Models:**

    - [Download Pretrained Weights](https://drive.google.com/file/d/1Rqty9pRhGdfhkfqlWbFUFgdFp0DvfORN/view)
    - Place the models in the `pretrained` directory.

4. **Organize Test Datasets:**

    ```
    <dataset_root>
    ├── original/        # Original images
    └── labels/          # Corresponding ground truth masks
    ```

### How to Run the Evaluation

To run the evaluation on a specific OoD dataset, use the following command:

```bash
python eval.py --ood_dataset_path your_dataset_path/SML_Fishyscapes/Static --ood_dataset_name SML_Fishyscapes_Static --score_mode max_logit
```

#### Key Arguments

- `--ood_dataset_path`: Path to the directory containing the OoD dataset.
- `--ood_dataset_name`: Name of the OoD dataset (`SML_Fishyscapes_Static`, `SML_Fishyscapes_LostAndFound`, or `road_anomaly`).
- `--score_mode`: Method for calculating the anomaly score (`max_logit` or `standardized_max_logit`).

### Visualization

Visualization outputs are saved in the `segmentation_results` and `anomaly_maps` folders within the specified dataset directory.

#### Visualization Includes:

- **Segmentation Results:** Visualizes the predicted class labels.
- **Anomaly Heatmap:** Displays the normalized anomaly scores.
- **Anomaly Binary Map:** Shows a binary map of anomalies based on a specified threshold.

#### Directory Structure for Visualization Results:

```
<ood_dataset_path>
├── segmentation_results/
│   └── image_name_max_logit_segmentation.png
├── anomaly_maps/
│   ├── image_name_max_logit_anomaly_map.png
│   └── image_name_max_logit_anomaly_binary.png
```

### Evaluation Metrics

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

### Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{Jung_2021_ICCV,
    author    = {Jung, Sanghun and Lee, Jungsoo and Gwak, Daehoon and Choi, Sungha and Choo, Jaegul},
    title     = {Standardized Max Logits: A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
}
```

## ATTA

Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation.

### Setup and Dataset

#### Pretrained Weights

You can download the pretrained weights from the following link:

[Download Pretrained Weights](https://drive.google.com/file/d/1Rqty9pRhGdfhkfqlWbFUFgdFp0DvfORN/view)

#### Dataset

Download the test dataset from:

[Download Test Dataset](http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar)

**Number of Test Data Samples:**

- **RoadAnomaly:** 60
- **Fishyscapes_Static:** 30
- **Fishyscapes_LostAndFound:** 100

### How to Implement ATTA with ResNet101 Backbone

1. **Navigate to the ATTA Directory:**

    ```bash
    cd ATTA
    ```

2. **Run the Experiments:**

    ```bash
    ./run_experiments.sh
    ```

#### Key Arguments

- `--cfg`: Configuration file for ATTA implementation, set to `exp/atta.yaml`.
- `--dataset`: Dataset name (e.g., `RoadAnomaly`, `FS_Static`, `FS_LostAndFound`).
- `--method`: Applicable methods (e.g., `Max_logit`, `Energy`, `Standardized_max_logit`).

### Visualization

Visualization outputs are saved within the `ATTA/visualize/(Dataset_name)/(Method_name)` directory. Types of visualizations include:

- **label:** Displays the true label.
- **atta_anomaly:** Shows anomalies in white.
- **anomaly_score:** Displays an anomaly score heatmap.
- **anomaly_class:** Shows anomaly segmentation based on a specified threshold.

### Explanation

The original ATTA code lacks support for the ResNet101 backbone, so additional code was added to ensure compatibility, including saving an in-class variable in `global_vars.py`. Visualization functions are available in `visualize.py`, and the Standardized Max-Logit method is applied within ATTA.

### Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{
    gao2023atta,
    title={{ATTA}: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation},
    author={Zhitong Gao and Shipeng Yan and Xuming He},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=bGcdjXrU2w}
}
```

## Combined Usage

To utilize both **Standardized Max Logits** and **ATTA** within the **SML_ATTA** framework, follow the setup and installation steps outlined above. You can independently run evaluations and visualizations for each method as needed. Combining insights from both approaches can lead to more robust OoD detection performance.

### Example Workflow

1. **Run Standardized Max Logits Evaluation:**

    ```bash
    cd standardized-max-logits
    python eval.py --ood_dataset_path your_dataset_path/SML_Fishyscapes/Static --ood_dataset_name SML_Fishyscapes_Static --score_mode standardized_max_logit
    ```

2. **Run ATTA Experiments:**

    ```bash
    cd ATTA
    ./run_experiments.sh
    ```

3. **Compare Results:**

    Analyze the evaluation metrics and visualization outputs from both methods to determine their effectiveness on your specific dataset.
