# CNN Architecture Benchmark on CIFAR-10

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-17/VGG16-vs-ResNet50-vs-EfficientNet-B0-A-Comparative-Study/blob/main/VGG16_vs_ResNet50_vs_EfficientNet_B0_comparative_study.ipynb)

A comparative benchmark of **VGG16**, **ResNet50**, and **EfficientNet-B0** fine-tuned from ImageNet-pretrained weights on the CIFAR-10 dataset. The benchmark reports test accuracy, parameter count, training time, and inference latency, and includes Grad-CAM visualizations for explainability.

## Contents

| File | Description |
|------|-------------|
| `CNN_Benchmark_CIFAR10.ipynb` | Main Colab notebook |
| `Comparative Study Report.pdf` | IEEE-style report summarizing the results |

## Results

| Model | Params (M) | Training Time (s) | Inference (ms/img) | Test Accuracy (%) |
|---|---:|---:|---:|---:|
| VGG16 | 134.30 | 1488 | 1.847 | 93.55 |
| ResNet50 | 23.53 | 598 | 0.905 | **96.51** |
| EfficientNet-B0 | **4.02** | 905 | **0.610** | 95.88 |

ResNet50 achieves the highest accuracy. EfficientNet-B0 reaches near-equal accuracy with ~33× fewer parameters than VGG16 and the lowest inference latency, making it the best efficiency choice. VGG16 is Pareto-dominated.

## Running the Notebook

1. Click the **Open in Colab** badge above.
2. Set the runtime to **GPU** (`Runtime → Change runtime type → GPU`).
3. Run the cells top to bottom. The notebook mounts Google Drive and saves trained models to `MyDrive/my_colab_models/CNNs/`.

### Caching / Re-running

Two flags at the top of the notebook control training behavior:

- `TRAIN_MODELS` — master switch. If a checkpoint already exists on Drive, the model is **loaded** and not retrained. Training only runs when no checkpoint is found.
- `FORCE_RETRAIN` — set to `True` to retrain a model even when a checkpoint exists.

Each training run saves a checkpoint with the best epoch and validation loss in its filename, e.g. `ResNet50_ep7_vl0.3210.pth`, so multiple runs accumulate distinct files and the best one (lowest val_loss) is auto-loaded.

## Pipeline

- **Dataset:** CIFAR-10 with a stratified 45k / 5k / 10k train-val-test split, resized to 128×128, normalized with ImageNet statistics, and augmented (random crop + horizontal flip).
- **Training:** Full fine-tuning with Adam (lr = 1e-4), batch size 64, mixed precision, up to 15 epochs with early stopping (patience 4) on validation loss.
- **Evaluation:** Accuracy, precision, recall, F1-score, per-class confusion matrix, and per-image inference latency on the test set.
- **Explainability:** Grad-CAM heatmaps generated from the last convolutional block of each model.

## Author

Aviral Singh
