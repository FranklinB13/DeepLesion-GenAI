i # How To Run This Project — Step by Step

This project uses **Variational Autoencoders (VAE)** to analyze lung lesion CT scans from the LIDC-IDRI dataset and predict malignancy. It is based on this research paper: https://arxiv.org/abs/2311.15719

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install Dependencies](#2-install-dependencies)
3. [Get the Dataset](#3-get-the-dataset)
4. [Preprocessing](#4-preprocessing)
5. [Train a VAE Model](#5-train-a-vae-model)
6. [Extract Latent Vectors](#6-extract-latent-vectors)
7. [Classify with MLP](#7-classify-with-mlp)
8. [Clustering and Visualization](#8-clustering-and-visualization)
9. [Understanding the Output Files](#9-understanding-the-output-files)
10. [Project Structure Overview](#10-project-structure-overview)

---

## 1. Prerequisites

Before starting, make sure you have the following installed on your machine:

- **Python 3.8 or higher** — [Download here](https://www.python.org/downloads/)
- **pip** — comes with Python
- **Jupyter Notebook** — to run `.ipynb` files
- A **GPU** is strongly recommended for training (CUDA-compatible NVIDIA GPU). Training on CPU is possible but very slow.

To check if Python is installed, open a terminal and run:

```bash
python --version
```

---

## 2. Install Dependencies

There is no `requirements.txt` file, so you need to install the libraries manually. Run the following command in your terminal:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchsummary pytorch-msssim
pip install numpy pandas scikit-learn scipy matplotlib seaborn pillow
pip install jupyter notebook classix-clustering
```

> **Note:** If you do not have a GPU, replace the first line with:
> ```bash
> pip install torch torchvision torchaudio
> ```

To verify PyTorch installed correctly and can detect your GPU:

```bash
python -c "import torch; print(torch.__version__); print('GPU available:', torch.cuda.is_available())"
```

---

## 3. Get the Dataset

This project uses the **LIDC-IDRI** dataset, a public dataset of lung CT scans.

1. Go to: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254
2. Create a free account and download the dataset (it is large — several hundred GB).
3. Also download the associated **XML annotation files** which contain the lesion segmentation masks.

> **Important:** The preprocessing scripts expect the raw DICOM files and XML annotations to be available. Make sure you know the path to where you downloaded them.

---

## 4. Preprocessing

The preprocessing is done through **Jupyter Notebooks**, which you run step by step. Open a terminal in the project root and launch Jupyter:

```bash
jupyter notebook
```

Then run the following notebooks **in order**:

### Step 4.1 — Convert DICOM to Numpy

Open: `Preprocessing/LIDC_DICOM_to_Numpy.ipynb`

- **What it does:** Reads the raw CT scan DICOM files and converts them to numpy arrays. Crops each lesion image to a 64×64 pixel region of interest (ROI).
- **Before running:** Update the path to your downloaded DICOM files inside the notebook. Look for a variable like `IMAGE_DIR` or similar and point it to your local dataset.
- **Output:** Numpy arrays (`.npy` files) of the cropped lesion images.

### Step 4.2 — Prepare Malignancy Labels

Open: `Preprocessing/LIDC_datasplit.ipynb`

- **What it does:** Reads the LIDC metadata and extracts malignancy scores (rated 1–5 by radiologists). Saves labels as numpy arrays. Removes excluded/ambiguous slices.
- **Output:** Label arrays (`.npy` files).

### Step 4.3 — Analyze ROI Size (Optional)

Open: `Preprocessing/mask_size.ipynb`

- **What it does:** Computes the optimal region of interest size using convex hull analysis on the segmentation masks. This is mostly for reference/understanding — you can skip this if you trust the 64×64 crop size.

### Step 4.4 — Create Train/Validation/Test Splits

Open: `Preprocessing/Train_Test_Split.ipynb`

- **What it does:** Splits patients (not individual slices) into training, validation, and test sets. This is important to avoid data leakage.
- **Output:** Two CSV metadata files saved in `Preprocessing/`:
  - `meta_mal_ben.csv` — malignant vs benign split (excludes ambiguous cases)
  - `meta_mal_nonmal.csv` — malignant vs non-malignant split (includes ambiguous as non-malignant)

---

## 5. Train a VAE Model

Training is done via **Python scripts** in the `VAE/` directory. You run them from the terminal.

### Choose your model

There are 5 model scripts. Start with the simplest one:

| Script | Description | Recommended for |
|--------|-------------|-----------------|
| `RandomSearchVAE.py` | Gaussian VAE + hyperparameter search | **Start here** |
| `RandomSearch_Dirichlet_VAE.py` | Dirichlet VAE + hyperparameter search | Better disentanglement |
| `VAE_joint_loss_mal_benign.py` | Gaussian VAE with joint classification loss (mal vs ben) | End-to-end learning |
| `VAE_MLP_joint_loss_mal_nonmal.py` | Gaussian VAE with joint classification loss (mal vs non-mal) | End-to-end learning |
| `VAE_Dirichlet_joint_loss.py` | Dirichlet VAE with joint classification loss | Best of both |

### Before running — update paths

Open the script you want to run in a text editor and look for these variables near the top. Change them to match your local setup:

```python
IMAGE_DIR = "path/to/your/images"           # Where your numpy image files are
results_path = "path/to/save/results"       # Where to save model checkpoints
meta_location = "Preprocessing/"            # Path to the metadata CSV files
```

> The original scripts have HPC (supercomputer) paths like `/nobackup/mm17b2k/...`. Replace these with paths on your own machine.

### Run the training

```bash
# Example: train the Gaussian VAE with random hyperparameter search
python VAE/RandomSearchVAE.py
```

- Training will iterate through random combinations of hyperparameters and print results.
- Model weights are saved automatically to the `results_path` you configured.
- Training time depends heavily on your hardware. On a GPU, expect several hours.

---

## 6. Extract Latent Vectors

After training, you need to extract the **latent vectors** (the compressed representations learned by the VAE).

Open: `VAE/Extract Latent Vectors and Reconstructions.ipynb`

- **What it does:** Loads the saved VAE model weights and passes all images through the encoder to get their latent vectors. Also generates reconstructed images for visual inspection.
- **Before running:** Update the path to your saved model weights file (`.pt` file).
- **Output:** Numpy arrays of latent vectors, saved for use in the next steps.

---

## 7. Classify with MLP

Once you have latent vectors, you can train a simple **MLP (Multi-Layer Perceptron)** classifier on top of them to predict malignancy.

### Run the hyperparameter search

```bash
# For Gaussian VAE latent vectors:
python "Clustering and MLP/RandomSearchMLP.py"

# For Dirichlet VAE latent vectors:
python "Clustering and MLP/Dirichlet_RandomSearchMLP.py"
```

- **Before running:** Update the path to your latent vector files in the script.
- **What it does:** Performs cross-validation with many MLP configurations to find the best classifier. Outputs metrics: Accuracy, Precision, Recall, Specificity, F1 Score, AUC.

---

## 8. Clustering and Visualization

These notebooks let you visually explore the latent space and cluster the learned representations.

### Step 8.1 — Initial Exploration

Open: `Clustering and MLP/Clustering_initial.ipynb`

- Loads latent vectors and visualizes them using **PCA** and **t-SNE**.
- Runs basic **K-Means** clustering.
- Good starting point to understand the latent space structure.

### Step 8.2 — Optimized Clustering

Open: `Clustering and MLP/Clustering.ipynb`

- Grid search for optimal clustering parameters.
- Compares **K-Means** and **CLASSIX** clustering algorithms.

### Step 8.3 — Latent Space Traversal

Open: `Clustering and MLP/Exploration_gaussian.ipynb`

- Traverses each dimension of the latent space to understand what features it encodes.
- Generates **GIF animations** showing how changing one latent dimension affects the reconstructed image.
- Example GIFs are already in the repo: `Ex1.gif` through `Ex5.gif`.

---

## 9. Understanding the Output Files

| File/Directory | Description |
|----------------|-------------|
| `Preprocessing/meta_mal_ben.csv` | Metadata for malignant vs benign split |
| `Preprocessing/meta_mal_nonmal.csv` | Metadata for malignant vs non-malignant split |
| `data_2classes.npy` | Image data for 2-class problem |
| `data_3classes.npy` | Image data for 3-class problem |
| `labels2.npy` | Labels for 2-class problem |
| `labels3.npy` | Labels for 3-class problem |
| `Ex1.gif` – `Ex5.gif` | Example latent traversal animations |
| `VAE_hyperparam_training.xlsx` | Spreadsheet tracking hyperparameter search results |

---

## 10. Project Structure Overview

```
Projet_GenAI/
│
├── Preprocessing/              # Step 1: Data preparation notebooks
│   ├── LIDC_DICOM_to_Numpy.ipynb
│   ├── LIDC_datasplit.ipynb
│   ├── mask_size.ipynb
│   ├── Train_Test_Split.ipynb
│   ├── meta_mal_ben.csv
│   └── meta_mal_nonmal.csv
│
├── VAE/                        # Step 2: VAE model training scripts
│   ├── RandomSearchVAE.py
│   ├── RandomSearch_Dirichlet_VAE.py
│   ├── VAE_joint_loss_mal_benign.py
│   ├── VAE_MLP_joint_loss_mal_nonmal.py
│   ├── VAE_Dirichlet_joint_loss.py
│   └── Extract Latent Vectors and Reconstructions.ipynb
│
├── Clustering and MLP/         # Step 3: Classification and analysis
│   ├── Clustering_initial.ipynb
│   ├── Clustering.ipynb
│   ├── Exploration_gaussian.ipynb
│   ├── RandomSearchMLP.py
│   └── Dirichlet_RandomSearchMLP.py
│
├── interesting_examples/       # Example output GIFs
├── Ex1.gif – Ex5.gif           # Latent traversal visualizations
├── data_2classes.npy           # Image data arrays
├── data_3classes.npy
├── labels2.npy
├── labels3.npy
├── README.md
└── HowTo.md                    # This file
```

---

## Quick Summary — Full Pipeline

```
1. Download LIDC-IDRI dataset
        ↓
2. Run Preprocessing notebooks (in order)
        ↓
3. Train a VAE model (e.g., RandomSearchVAE.py)
        ↓
4. Extract latent vectors (Extract Latent Vectors notebook)
        ↓
5. Run MLP classification (RandomSearchMLP.py)
        ↓
6. Explore results (Clustering and Exploration notebooks)
```

---

## Common Issues

**"CUDA is not available"**
Your GPU is not detected. Either install the CUDA version of PyTorch (see Step 2) or the code will automatically fall back to CPU (slow but functional).

**"FileNotFoundError" when running scripts**
You need to update the hardcoded paths in the scripts to match where your files are saved on your machine.

**Jupyter notebook won't open**
Make sure Jupyter is installed (`pip install notebook`) and run `jupyter notebook` from the project root directory.

**Out of memory error during training**
Reduce the `batch_size` hyperparameter in the script. Try values like 32 or 64.
