# ATTA

Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation.

## Table of Contents

- [Setup and Dataset](#setup-and-dataset)
  - [Pretrained Weights](#pretrained-weights)
  - [Dataset](#dataset)
- [How to Implement ATTA with ResNet101 Backbone](#how-to-implement-atta-with-resnet101-backbone)
  - [Key Arguments](#key-arguments)
- [Visualization](#visualization)
- [Explanation](#explanation)
- [Citation](#citation)

## Setup and Dataset

### Pretrained Weights

You can download the pretrained weights from the following link:

[Download Pretrained Weights](https://drive.google.com/file/d/1Rqty9pRhGdfhkfqlWbFUFgdFp0DvfORN/view)

### Dataset

Download the test dataset from:

[Download Test Dataset](http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar)

**Number of Test Data Samples:**

- **RoadAnomaly:** 60
- **Fishyscapes_Static:** 30
- **Fishyscapes_LostAndFound:** 100

## How to Implement ATTA with ResNet101 Backbone

1. **Navigate to the ATTA directory:**

    ```bash
    cd ATTA
    ```

2. **Run the experiments:**

    ```bash
    ./run_experiments.sh
    ```

### Key Arguments

- `--cfg`: Configuration file for ATTA implementation, set to `exp/atta.yaml`.
- `--dataset`: Dataset name (e.g., `RoadAnomaly`, `FS_Static`, `FS_LostAndFound`).
- `--method`: Applicable methods (e.g., `Max_logit`, `Energy`, `Standardized_max_logit`).

## Visualization

Visualization outputs are saved within the `ATTA/visualize/(Dataset_name)/(Method_name)` directory. Types of visualizations include:

- **label:** Displays the true label.
- **atta_anomaly:** Shows anomalies in white.
- **anomaly_score:** Displays an anomaly score heatmap.
- **anomaly_class:** Shows anomaly segmentation based on a specified threshold.

## Explanation

The original ATTA code lacks support for the ResNet101 backbone, so additional code was added to ensure compatibility, including saving an in-class variable in `global_vars.py`. Visualization functions are available in `visualize.py`, and the Standardized Max-Logit method is applied within ATTA.

## Citation

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
