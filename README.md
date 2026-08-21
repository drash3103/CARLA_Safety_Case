# CARLA Out-of-Distribution (OOD) Detection

## Project Overview

This repository contains the implementation for training and evaluating binary image classifiers on a CARLA-simulated autonomous-driving dataset. Three independent models detect the presence of pedestrians, vehicles, and traffic lights.

The repository also contains experiments for in-distribution evaluation, FGSM adversarial robustness, calibration, Grad-CAM visualisation, and out-of-distribution (OOD) detection under conditions such as fog and night.

The main implementation is provided as a Google Colab notebook.

## Dataset Setup

The CARLA dataset is **not included in this repository**.

After downloading the dataset, extract it so that the structure is:

```text
extracted_dataset/
├── train/
│   ├── labels.csv
│   └── rgb-front/
├── validation/
│   ├── labels.csv
│   └── rgb-front/
├── test/
│   ├── labels.csv
│   └── rgb-front/
├── test-fog/
│   ├── labels.csv
│   └── rgb-front/
├── test-night/
│   ├── labels.csv
│   └── rgb-front/
└── test-town-01/
    ├── labels.csv
    └── rgb-front/
```

Set the dataset path in the notebook, for example:

```python
DATASET_PATH = "/content/drive/MyDrive/2026/extracted_dataset"
```

## Environment and Dependencies

Main dependencies:

- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- tqdm

A GPU is recommended for training and feature extraction.

## Reproducing the Experiments

The recommended way to reproduce the results is to open the provided Colab notebook, set the dataset path, and execute the relevant cells.

The main experiments are:

1. Binary detector training
2. In-distribution evaluation
3. FGSM adversarial robustness
4. Calibration and temperature scaling
5. OOD detection
6. Grad-CAM visualisation

## 1. Training the Binary Classifiers

Three independent binary classifiers are trained:

```text
Pedestrian      → has_pedestrian
Vehicle         → has_vehicle
Traffic light   → has_traffic_light
```

The classifiers use an ImageNet-pretrained ResNet-18 backbone with a binary classification head.

Training includes image resizing, data augmentation, ImageNet normalization, BCEWithLogitsLoss, Adam optimization, learning-rate scheduling, and early stopping.

To retrain a model:

1. Open the notebook in Google Colab.
2. Mount Google Drive if required.
3. Set `DATASET_PATH`.
4. Run the dataset and preprocessing cells.
5. Run the training cell for the desired detector.
6. The best validation-loss checkpoint is saved as a `.pth` file.

## 2. Using the Trained Checkpoints

The trained `.pth` checkpoints can be used **without retraining**. This is the recommended approach when reproducing the evaluation results.

A checkpoint contains the trained model parameters. The model architecture must match the architecture used when the checkpoint was created.

### Loading a checkpoint

Use the same model definition used during training and then load the checkpoint:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BinaryClassifier(pretrained=False)

model.load_state_dict(
    torch.load(
        "/path/to/model_checkpoint.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()
```

Replace the checkpoint path with the location of the required `.pth` file, for example:

```python
model_path = "/content/drive/MyDrive/Models/vehicle_model.pth"
```

If the checkpoint was trained with a modified architecture, that exact architecture must be recreated before loading the checkpoint.

### Using the checkpoint for evaluation

After loading the model:

```python
model.eval()
```

the checkpoint can be passed directly to the evaluation cells in the notebook.

For binary predictions, the model output is converted to a probability using:

```python
probability = torch.sigmoid(output)
```

The evaluation code in the notebook can then be used to calculate recall, precision, accuracy, F1-score, calibration metrics, FGSM robustness, OOD AUROC, and Grad-CAM visualisations.

**Important:** The checkpoint, model architecture, preprocessing, and input image format should be kept consistent with the original experiment when reproducing the reported results.

## 3. In-Distribution Detection Performance

The binary classifiers are evaluated on the separate in-distribution test set using metrics including recall, precision, accuracy, and F1-score.

Run the corresponding evaluation section after loading the trained checkpoints.

## 4. FGSM Adversarial Robustness

The notebook evaluates robustness against the Fast Gradient Sign Method (FGSM) using the specified perturbation:

```text
epsilon = 0.05
```

The experiment compares clean performance with performance after applying the perturbation.

Run the FGSM evaluation cells after loading the desired checkpoint.

## 5. Calibration and Temperature Scaling

The notebook evaluates Expected Calibration Error (ECE) before and after temperature scaling.

The validation set is used for calibration and the test set is used for evaluation.

Run the calibration section after loading the corresponding checkpoint.

## 6. Out-of-Distribution Detection

The repository includes feature-space OOD detection experiments for conditions including:

- Fog
- Night
- Different CARLA town/scenario data where applicable

For k-NN OOD detection, the workflow is:

```text
Trained classifier
       ↓
ResNet-18 feature extractor
       ↓
Feature representation
       ↓
k-NN fitted on in-distribution features
       ↓
Distance-based OOD score
       ↓
AUROC
```

Higher nearest-neighbour distance indicates that an input is further from the in-distribution feature space.

To reproduce the k-NN OOD results:

1. Load the trained checkpoint.
2. Extract features from the in-distribution reference data.
3. Fit the k-NN detector.
4. Extract features from the in-distribution test, Fog, and Night datasets.
5. Calculate distance-based OOD scores.
6. Calculate AUROC.

## 7. Grad-CAM Visualisation

Grad-CAM visualisations can be generated to inspect which image regions contribute to model predictions.

The notebook includes visualisation for different driving conditions, including in-distribution daytime and OOD scenarios.

## Limitations

The following limitations should be considered when interpreting the results:

- **Pedestrian detection performance:** The pedestrian classifier performs substantially worse than the vehicle and traffic-light classifiers on the in-distribution test set. Small, occluded, or visually ambiguous pedestrians are particularly challenging.

- **Potential annotation inconsistency:** Manual inspection identified at least one potential inconsistency between a provided label and the corresponding image. For example, frame `13400` is labelled `has_pedestrian = True`, although no clearly visible pedestrian can be identified in the corresponding RGB image. Such annotation issues may negatively affect training and evaluation, particularly for pedestrian detection.

- **Presence-only detection:** The models predict only whether a pedestrian, vehicle, or traffic light is present. The traffic-light detector does not determine the signal state (e.g., red or green), and the models do not provide object location or distance.

- **Adversarial robustness:** The models show substantial degradation under the evaluated FGSM perturbation with `epsilon = 0.05`. They should therefore not be considered robust against adversarial manipulation of the camera input.

- **OOD coverage:** The OOD experiments focus on the available Fog and Night scenarios. These conditions do not cover every possible unseen environment or weather condition.

- **Camera-only perception:** The evaluated perception system relies on the front-facing RGB camera. No independent LiDAR, radar, or depth sensor is used as a backup perception source.

- **Fallback validation:** A complete autonomous fallback and human-takeover mechanism is not experimentally validated in these experiments. Therefore, its effectiveness and response time cannot be established from the reported model experiments alone.

- **Model improvement trade-off:** An additional pedestrian-model improvement attempt increased recall but substantially reduced precision and overall accuracy. Thus, increasing model complexity alone did not provide a clear overall improvement.

These limitations should be considered when interpreting the reported performance and when extending the system.

## Notes

- The CARLA dataset is not included in this repository.
- A GPU is recommended for training and feature extraction.
- Dataset paths may need to be changed depending on the execution environment.
- Provided checkpoints can be used to reproduce evaluation results without retraining.
- The model architecture must match the architecture used to create a checkpoint.
- Small numerical differences may occur when models are retrained because of random initialization, library versions, and hardware/software differences.
- Training from scratch is optional when the corresponding trained checkpoints are available.

## Quick Reproduction

```text
1. Download and extract the CARLA dataset
              ↓
2. Open the Colab notebook
              ↓
3. Set DATASET_PATH
              ↓
4. Load the provided .pth checkpoints
              ↓
5. Run the evaluation sections
              ↓
6. Reproduce detection, robustness,
   calibration, OOD and Grad-CAM results
```

Training from scratch is only required if the checkpoints are not available or if you want to retrain the models.
