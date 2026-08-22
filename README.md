# CARLA OOD Detection and Safety Evaluation

## Project Overview

This repository contains three independent binary image classifiers for
pedestrian, vehicle, and traffic-light detection on a CARLA-simulated
driving dataset. It also includes experiments for in-distribution
performance, FGSM robustness, calibration, feature-space OOD detection,
Grad-CAM explanations, and ODD k-projection coverage.

The main implementation is provided as a Google Colab notebook.

## Dataset Setup

The CARLA dataset is not included in this repository. Extract it with
the following structure:

``` text
extracted_dataset/
├── train/           ├── labels.csv
│                   └── rgb-front/
├── validation/      ├── labels.csv
│                   └── rgb-front/
├── test/            ├── labels.csv
│                   └── rgb-front/
├── test-fog/        ├── labels.csv
│                   └── rgb-front/
├── test-night/      ├── labels.csv
│                   └── rgb-front/
└── test-town-01/    ├── labels.csv
                    └── rgb-front/
```

Set the path in the notebook:

``` python
DATASET_PATH = "/content/drive/MyDrive/2026/extracted_dataset"
```

## Environment

Main dependencies: PyTorch, Torchvision, NumPy, Pandas, Scikit-learn,
Matplotlib, Pillow, and tqdm. A GPU is recommended for training and
feature extraction.

# Reproducing the Results

## 1. Binary Detector Training (Run exercise 3 cells)

Three independent classifiers are trained:

``` text
Pedestrian      → has_pedestrian
Vehicle         → has_vehicle
Traffic light   → has_traffic_light
```

The models use an ImageNet-pretrained ResNet-18 backbone with a binary
classification head.

To reproduce training:

1.  Open the Colab notebook.
2.  Install/import the required dependencies.
3.  Mount Google Drive if required.
4.  Set `DATASET_PATH`.
5.  Run the dataset and preprocessing cells.
6.  Run the training cell for the required detector.
7.  The best validation-loss checkpoint is saved as a `.pth` file.

## 2. Using the Trained Checkpoints

Retraining is not required when the provided checkpoints are available.

Use the same model definition and preprocessing used during training:

``` python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BinaryClassifier(pretrained=False)
model.load_state_dict(
    torch.load("/path/to/model_checkpoint.pth", map_location=device)
)
model = model.to(device)
model.eval()
```

If a checkpoint was created with a modified architecture, recreate that
exact architecture before loading it.

## 3. In-Distribution Detection Performance (Run Exercise 3.6 cells)

1.  Load the required checkpoint.
2.  Use the original test-set preprocessing.
3.  Run the in-distribution evaluation cells.
4.  Calculate accuracy, precision, recall, and F1-score.

The safety-case analysis primarily uses per-class recall and the
specified detection thresholds.

## 4. FGSM Adversarial Robustness (Run Exercise 8.5 cell)

The reported experiment uses:

``` python
epsilon = 0.05
```

To reproduce:

1.  Load a trained detector.
2.  Evaluate clean test images.
3.  Generate FGSM-perturbed images with `epsilon = 0.05`.
4.  Re-evaluate the detector.
5.  Compare clean and adversarial recall and calculate the recall drop.

## 5. Calibration and Temperature Scaling (Run Exercises 7.4 and 7.5 cells)

1.  Load the trained checkpoint.
2.  Use the validation set to estimate the temperature parameter.
3.  Apply temperature scaling to the logits.
4.  Evaluate on the separate test set.
5.  Calculate ECE before and after scaling.

The validation set is used for calibration and the test set for final
evaluation.

## 6. Feature-Space OOD Detection (Run Exercise 9.7 cell)

OOD detection uses ResNet-18 feature representations and a
 distance score for available conditions such as Fog,
Night, and different CARLA town/scenario data.

To reproduce:

1.  Load the trained checkpoint.
2.  Extract ResNet-18 features from the in-distribution reference data.
3.  Use Mahalanobis.
4.  Extract features from ID and OOD test sets.
5.  Calculate AUROC.



## 7. Grad-CAM Visualisation (Run Exercise 6.5 cell)

1.  Load the required checkpoint.
2.  Select representative ID and OOD images.
3.  Run the Grad-CAM cell using the same target layer.
4.  Generate the heatmap overlays.
5.  Compare daytime ID, night-time OOD, and different-town OOD examples.

Grad-CAM provides qualitative evidence about model attention and does
not establish prediction correctness by itself.

## 8. ODD k-Projection Coverage (Run Exercise 4.5 cell)

ODD coverage evaluates how well combinations of ODD dimensions are
represented in the test scenarios. The analysis uses:

``` text
Weather:  sunny, fog
Lighting: day, night
Town:     default, town01
```


To reproduce:

1.  Download the `odd-coverage` implementation.
2.  Define the ODD dimensions and possible values.
3.  Encode the available test scenarios.
4.  Run k-projection coverage for `k = 1, 2, 3`.
5.  Record the coverage values.

Example:

``` python
scenarios = [
    {"weather": "sunny", "lighting": "day",   "town": "default"},
    {"weather": "fog",   "lighting": "day",   "town": "default"},
    {"weather": "sunny", "lighting": "night", "town": "default"},
    {"weather": "sunny", "lighting": "day",   "town": "town01"},
]
```

## Complete Reproduction Workflow

``` text
CARLA dataset
      ↓
Open Colab notebook
      ↓
Set DATASET_PATH
      ↓
Load checkpoints
      ↓
In-distribution evaluation
      ↓
FGSM robustness
      ↓
Calibration / temperature scaling
      ↓
Feature-space OOD detection
      ↓
Grad-CAM
      ↓
ODD k-projection coverage
```

Training from scratch is only required if the checkpoints are
unavailable or retraining is desired.

## Reproducibility Notes

-   Keep the checkpoint architecture unchanged.
-   Use the same preprocessing and normalization.
-   Use the same dataset splits.
-   Use `epsilon = 0.05` for the reported FGSM experiment.
-   Use validation data for temperature scaling and test data for final
    calibration evaluation.
-   Use the same reference features and k-NN configuration for OOD
    detection.
-   Use the same ODD dimensions and scenario encoding for k-projection
    coverage.
-   Small numerical differences may occur due to hardware, library
    versions, random seeds, or retraining.
-   The CARLA dataset is not included in this repository.

## Limitations

The pedestrian detector performs substantially worse than the vehicle
and traffic-light detectors. The models also degrade under the evaluated
FGSM perturbation, while OOD experiments cover only the available
scenarios. The classifiers perform presence-only detection and do not
provide object location, distance, or traffic-light state. The reported
experiments therefore provide evidence for the evaluated conditions but
do not establish unrestricted deployment safety.
