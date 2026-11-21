# SemiFedFlood

Semi-supervised federated learning for flood-area segmentation.

This repository contains the code, dataset, architecture diagram, and project documentation for a controlled comparison of centralized and federated flood segmentation models. The primary experiment notebook is `Full_Run.ipynb`.

## Repository contents

- `Full_Run.ipynb` - complete experiment pipeline for the main controlled run
- `D3_Code_01.ipynb` - earlier U-Net experiment code
- `D3-Code-02.ipynb` - earlier ResNet-50 experiment code
- `flood_area/` - image and mask dataset
- `architecure1.png` - project architecture diagram
- `D1-Introduction-and-Literature-Review.pdf`
- `D2-Dataset-Description-and-Source.pdf`
- `D2-Methodology and Experimental Setup.pdf`
- `D3-Experiments-and-Results.pdf`

## Experiment setup

The main notebook compares:

- Centralized ResNet-50 training
- Federated Averaging (FedAvg)
- FedAvg with 20% labeled data
- FedAvg with student-teacher semi-supervised learning
- FedProx with 20% labeled data
- FedProx with student-teacher semi-supervised learning

Training and evaluation use common image/mask corrections, train-validation-test splits, client assignments, label selections, batch budgets, and segmentation thresholds.

## Run in Google Colab

1. Open `Full_Run.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Upload `flood_area.zip` to:

   `/content/drive/MyDrive/DLP_Project/flood_area.zip`

4. Run all notebook cells.

Checkpoints, metrics, plots, predictions, and error maps are saved under a timestamped output directory in Google Drive.

## Run with the repository dataset

The repository includes the extracted dataset at `flood_area/`. The notebook is configured for Colab and expects the archive at the Drive path above. If adapting it to a local or alternate environment, update `ZIP_PATH`, `DATA_ROOT_LOCAL`, `CACHE_ROOT`, and `PROJECT` near the start of the notebook.

Dataset mapping is defined in `flood_area/metadata.csv`.

## Requirements

The main notebook is designed for Python 3 with PyTorch and CUDA support. Install the core dependencies with:

```bash
pip install -r requirements.txt
```

Recommended runtime:

- Python 3
- NVIDIA GPU with CUDA
- PyTorch and torchvision
- Google Colab for the provided end-to-end workflow

## Reproducibility notes

The notebook uses controlled seeds and preserves architecture-specific normalization:

- ResNet-50 uses ImageNet-standardized RGB inputs.
- U-Net-style models use `[0,1]` RGB inputs.

The dataset split, client allocation, labeled subset selection, and evaluation protocol are saved for each run.

## Citation

If this work is used in a publication or project, cite the associated research paper and this repository.
