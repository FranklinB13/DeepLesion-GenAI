# Project Guide — Can a Dirichlet VAE Simultaneously Learn Organ and Malignancy?

**Group:** Bezet Camille, Brasa Franklin, Kenmogne Loïc, Martins Soares Flavio
**Course:** Deep Generative Models — Spring 2026
**Based on:** Keel et al. (2023) — arXiv:2311.15719

---

## What Are We Actually Trying to Do? (Plain English)

Before diving into code, let's understand the big picture.

A **Variational Autoencoder (VAE)** is a neural network that learns to compress images into a small set of numbers (called a **latent vector**), and then reconstruct them back. Think of it as teaching a machine to summarize what it sees in very few numbers.

The **Dirichlet VAE (DirVAE)** is a special kind of VAE that theoretically pushes each of those numbers to encode a *different, independent feature*. For example, one number might represent "how irregular the lesion border is", and another might represent "how dense the lesion appears on the scan" — without anyone explicitly telling it to do that.

**Our experiment:** We train both a standard VAE (GVAE) and a DirVAE on 32,735 CT scan patches from the NIH DeepLesion dataset. Crucially, 23,000 of those patches have **no labels at all** — the model only sees the images. After training, we take 9,000 patches that *do* have organ labels (lung, liver, kidney, etc.) and ask: **did the model's internal representation spontaneously organize by organ type AND by malignancy, without ever being told to?**

If yes → the DirVAE is genuinely learning clinical structure from visual patterns alone.
If no → that's also a valuable scientific finding worth publishing.

---

## Table of Contents

1. [Conceptual Background](#1-conceptual-background)
2. [Environment Setup](#2-environment-setup)
3. [Getting the DeepLesion Dataset](#3-getting-the-deeplesion-dataset-zero-local-storage-required)
4. [Understand the Dataset Structure](#4-understand-the-dataset-structure)
5. [Stage 1 — Replication on Lung-Only Data](#5-stage-1--replication-on-lung-only-data)
6. [Stage 2 — Full Unsupervised Training](#6-stage-2--full-unsupervised-training)
7. [Stage 3 — Dual-Axis Latent Traversals](#7-stage-3--dual-axis-latent-traversals)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [The Key Figure — UMAP Visualization](#9-the-key-figure--umap-visualization)
10. [Project File Structure](#10-project-file-structure)
11. [Timeline and Division of Work](#11-timeline-and-division-of-work)
12. [Glossary](#12-glossary)

---

## 1. Conceptual Background

### What is a Latent Space?

Imagine you compress a 64×64 grayscale image (4,096 numbers) into just 32 numbers. Those 32 numbers are the **latent vector** — a compact code that the model uses to represent the image. The collection of all possible such codes is the **latent space**.

The hope is that images that look similar (e.g., two lung lesions) end up close together in this space, and very different images (lung lesion vs. kidney lesion) end up far apart.

### GVAE vs DirVAE — What's the Difference?

| | GVAE (Gaussian VAE) | DirVAE (Dirichlet VAE) |
|---|---|---|
| **Latent prior** | Normal distribution N(0,1) | Dirichlet distribution Dir(α) |
| **Latent values** | Any real number | Positive, sums to 1 (like probabilities) |
| **Structure** | Dimensions can correlate | Dimensions encouraged to be independent |
| **Disentanglement** | Weaker | Stronger (each dimension = different feature) |

The Dirichlet prior puts values on a **simplex** — imagine a triangle where every point represents a mix of pure features. This geometry naturally encourages each latent dimension to specialize.

### What is Disentanglement?

A latent space is **disentangled** if:
- Changing dimension 1 only changes organ type in the reconstruction
- Changing dimension 2 only changes malignancy appearance
- These changes don't interfere with each other

This is what we're testing — does the DirVAE spontaneously achieve this without any supervision?

### The Probe Set Design

This is the smartest part of the experiment:

```
Training data (22,921 patches) ─── NO labels ──► VAE learns purely from images
                                                         ↓
                                              Learned latent space
                                                         ↓
Probe data (9,816 patches) ──── HAS organ labels ──► We project into the space
                                                         ↓
                              Measure: did clusters align with organs/malignancy?
```

The probe set is **never used during training**. It's only used afterwards to measure what the model learned. This is called **post-hoc evaluation** and it's the gold standard for testing unsupervised learning.

---

## 2. Environment Setup

### 2.1 Install Python

Make sure you have Python 3.9 or 3.10. Check with:

```bash
python --version
```

We recommend using a **virtual environment** to avoid conflicts between projects:

```bash
# Create a virtual environment named "deeplesion_env"
python -m venv deeplesion_env

# Activate it (Mac/Linux):
source deeplesion_env/bin/activate

# Activate it (Windows):
deeplesion_env\Scripts\activate
```

You should see `(deeplesion_env)` at the start of your terminal prompt.

### 2.2 Install All Required Packages

Create a file called `requirements.txt` in your project folder with this content:

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
torchsummary
pytorch-msssim
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
pillow
jupyter
notebook
umap-learn
classix-clustering
tqdm
```

Then install everything:

```bash
# With GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Without GPU (CPU only — much slower):
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 2.3 Verify Your Installation

Run this in Python to check everything works:

```python
import torch
import numpy as np
import pandas as pd
import umap
from pytorch_msssim import ssim

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("All imports OK!")
```

---

## 3. Getting the DeepLesion Dataset (Zero Local Storage Required)

### 3.1 What is DeepLesion?

NIH DeepLesion is a public dataset of 32,735 CT lesion patches collected from real hospital scans. It was released by the National Institutes of Health (NIH) in 2018. The full dataset is ~220 GB — impossible to download on a laptop or store in a personal Google Drive.

Here are your actual options, ordered from best to worst for your situation.

---

### Strategy A — Kaggle (Best Option — Completely Free, No Storage Needed)

**Kaggle** is a data science platform that hosts the DeepLesion dataset publicly. You get:
- The dataset **already uploaded and mounted** — no download
- **Free GPU** (NVIDIA T4, 30 hours/week) built into the browser notebook
- **100 GB of free storage** per account
- Everything runs in the browser — nothing on your Mac

**Step 1: Create a free Kaggle account**

Go to https://www.kaggle.com and sign up.

**Step 2: Find the DeepLesion dataset on Kaggle**

Search for `NIH DeepLesion` on Kaggle or go directly to:
- Full subset: `kaggle.com/datasets/kmader/nih-deeplesion-subset`
- Tensor slices variant: `kaggle.com/datasets/benjaminahlbrecht/nih-deeplesion-tensor-slices`

**Step 3: Create a Kaggle Notebook**

Click "New Notebook" on the dataset page. This opens a Jupyter-like environment in your browser with the dataset already attached at `/kaggle/input/nih-deeplesion-subset/`.

**Step 4: Enable the free GPU**

In the notebook sidebar → Settings → Accelerator → select **GPU T4 x2**. Done. You now have a GPU and the data, all for free.

**Step 5: Point your code at the Kaggle input path**

```python
# In your Kaggle notebook, the dataset is already here — no download needed
IMAGE_DIR = "/kaggle/input/nih-deeplesion-subset/Images_png"
CSV_PATH  = "/kaggle/input/nih-deeplesion-subset/DL_info.csv"
```

**Step 6: Save your trained models to Kaggle output**

Kaggle notebooks save their outputs automatically. After training, download just the model weights file (a few hundred MB) to your Mac for analysis.

```python
# Save model weights — they will appear in /kaggle/working/ and be downloadable
torch.save(model.state_dict(), "/kaggle/working/dirvae_best.pt")
```

> **Tip:** Kaggle sessions last up to 12 hours. For long training runs, enable "Save & Run All" so the notebook runs even if you close your browser.

---

### Strategy B — HuggingFace Datasets with Streaming (No Download, Works Locally)

The dataset is also on HuggingFace Hub at `farrell236/DeepLesion`. HuggingFace supports **streaming**, meaning your code downloads individual images on-the-fly as needed — you never store the full dataset.

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Stream the dataset — images are fetched one by one, nothing stored locally
dataset = load_dataset(
    "farrell236/DeepLesion",
    split="train",
    streaming=True   # ← this is the key — no full download
)

# Iterate through batches
for batch in dataset.take(100):   # take() fetches only N samples
    image = batch["image"]        # PIL Image
    label = batch["label"]        # organ label
    # ... process and feed to your model
```

> **Downside:** Streaming is slower than reading from disk because each image is fetched over the internet. Fine for testing, but too slow for serious training. Use Kaggle for actual training.

**For fast local development: use the 2,000-sample balanced subset**

HuggingFace also hosts a tiny curated version with only 2,000 patches (one per organ, balanced):

```python
# Only ~200 MB — small enough to fully download and use locally
dataset = load_dataset("Voxel51/deeplesion-balanced-2k")
dataset.save_to_disk("deeplesion_small/")   # save locally for reuse
```

This is perfect for:
- Testing that your DataLoader works correctly
- Debugging your model architecture
- Running a quick training loop to check loss goes down

Switch to the full dataset on Kaggle only once your code is confirmed working.

---

### Strategy C — Ask Your School's HPC/Server

Many engineering schools have a High Performance Computer (HPC) cluster where large datasets like DeepLesion are already stored. The original paper (Keel et al.) was trained on an HPC — you can see the `/nobackup/` paths in the code.

Ask your professor or IT department:
> "Do you have the NIH DeepLesion dataset on the cluster? Can I get access to GPU nodes to train a PyTorch model?"

This is the best option for serious training if your school has it.

---

### Recommended Workflow for Your Group

```
Day 1-2:  Use HuggingFace "Voxel51/deeplesion-balanced-2k" (2k samples, ~200MB)
          → Write and debug your DataLoader locally on your Mac
          → Make sure images load, preprocessing works, model runs

Day 3+:   Move to Kaggle notebooks
          → Full dataset already there, free GPU
          → Run Stage 1 and Stage 2 training
          → Download only the .pt model weights files (small) when done
```

| Goal | Use |
|------|-----|
| Write and debug DataLoader | HuggingFace 2k subset — download locally |
| Train Stage 1 (lung only) | Kaggle notebook — free GPU, data already there |
| Train Stage 2 (full 32k) | Kaggle notebook — free GPU, data already there |
| Analysis and visualization | Run locally after downloading model weights |

---

### 3.2 Expected Folder Structure After Extraction

```
DeepLesion/
├── DL_info.csv              ← The main metadata file
└── Images_png/
    ├── 000001_01_01/        ← Patient folders
    │   ├── 000.png
    │   ├── 001.png
    │   └── ...
    ├── 000001_01_02/
    └── ...
```

Each folder is named `{patient_id}_{study_id}_{series_id}`. Inside are PNG images, one per CT slice.

---

## 4. Understand the Dataset Structure

### 4.1 Reading DL_info.csv

This CSV file is the heart of the dataset. Open it in Excel or Python to explore it. Key columns:

| Column | Description |
|--------|-------------|
| `File_name` | Path to the PNG image, e.g., `000001_01_01/007.png` |
| `Coarse_lesion_type` | Lesion type code (1–8 for organ, -1 for unknown) |
| `Train_Val_Test` | 1=Train, 2=Val, 3=Test |
| `Bounding_boxes` | Lesion bounding box in the image (x1, y1, x2, y2) |
| `RECIST_longest_diameter` | Size of the lesion |
| `Lesion_diameters_Pixel_` | Pixel dimensions |
| `Possibly_noisy` | Whether annotation might be noisy |
| `Slice_range` | Which slices the lesion spans |
| `Spacing_mm_px_` | Pixel spacing in mm |
| `Image_size` | Original image dimensions |
| `Key_slice_index` | The main slice index |
| `Patient_index` | Patient ID |
| `Study_index` | Study ID |
| `Series_ID` | Series ID |
| `Measurement_coordinates` | RECIST measurement coordinates |

### 4.2 The Organ Label Mapping

The `Coarse_lesion_type` column uses these codes:

| Code | Organ |
|------|-------|
| 1 | Bone |
| 2 | Abdomen |
| 3 | Mediastinum |
| 4 | Liver |
| 5 | Lung |
| 6 | Kidney |
| 7 | Soft Tissue |
| 8 | Pelvis |
| -1 | Unknown (no organ label) |

### 4.3 The Train/Val/Test Split

The `Train_Val_Test` column determines the role of each patch:

| Value | Role in Our Experiment |
|-------|------------------------|
| 1 (Train, ~22,921 patches) | **VAE training only** — model never sees organ labels |
| 2 (Val, ~4,817 patches) | **Probe set** — used only after training to evaluate latent space |
| 3 (Test, ~4,999 patches) | **Probe set** — used only after training to evaluate latent space |

> **Critical rule:** Never let the model see the organ labels during training. Only use Val and Test labels *after* training to probe the learned space.

### 4.4 Quick Data Exploration (run this first!)

Create a notebook called `00_data_exploration.ipynb` and run this:

```python
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the metadata
df = pd.read_csv("DeepLesion/DL_info.csv")

print(f"Total patches: {len(df)}")
print(f"\nTrain/Val/Test split:")
print(df['Train_Val_Test'].value_counts().sort_index())

print(f"\nOrgan label distribution:")
organ_names = {-1: 'Unknown', 1: 'Bone', 2: 'Abdomen', 3: 'Mediastinum',
               4: 'Liver', 5: 'Lung', 6: 'Kidney', 7: 'Soft Tissue', 8: 'Pelvis'}
print(df['Coarse_lesion_type'].map(organ_names).value_counts())

# Show how many labelled patches are in val/test
probe = df[df['Train_Val_Test'].isin([2, 3])]
print(f"\nProbe set (val+test): {len(probe)} patches")
print(f"Labelled in probe set: {(probe['Coarse_lesion_type'] != -1).sum()} patches")
```

---

## 5. Stage 1 — Replication on Lung-Only Data

**Goal:** Before tackling the full dataset, prove that our pipeline correctly reproduces the original paper's results on lung CT patches only.

### 5.1 Why Do This First?

If you skip Stage 1 and jump to Stage 2 and get bad results, you won't know if the problem is:
- Your data loading code is wrong
- Your preprocessing is wrong
- Your model training is wrong
- The experiment just didn't work

Stage 1 gives you a known baseline. If you can reproduce AUC ~0.98 on lung patches, your pipeline is correct.

### 5.2 Filter to Lung-Only Patches

Create a script `01_filter_lung.py`:

```python
import pandas as pd

df = pd.read_csv("DeepLesion/DL_info.csv")

# Filter: lung only (code = 5), from val/test (our probe set)
lung_probe = df[(df['Coarse_lesion_type'] == 5) & (df['Train_Val_Test'].isin([2, 3]))]

print(f"Lung patches in probe set: {len(lung_probe)}")
# Expected: ~2,394 patches

# For Stage 1, we also need lung patches from train for VAE training
lung_train = df[(df['Coarse_lesion_type'] == 5) & (df['Train_Val_Test'] == 1)]
print(f"Lung patches in train set: {len(lung_train)}")

# Save filtered metadata
lung_train.to_csv("data/stage1_lung_train.csv", index=False)
lung_probe.to_csv("data/stage1_lung_probe.csv", index=False)
```

### 5.3 Write the DeepLesion DataLoader

This is the main code change compared to the original repo. Create `deeplesion_dataset.py`:

```python
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class DeepLesionDataset(Dataset):
    """
    PyTorch Dataset for the NIH DeepLesion dataset.

    Returns preprocessed 64x64 grayscale CT patches.
    Preprocessing: HU clipping → normalize to [0,1] → crop to 64x64
    """

    def __init__(self, csv_path, images_dir, hu_min=-1000, hu_max=400, img_size=64):
        """
        Args:
            csv_path:   Path to the filtered DL_info.csv
            images_dir: Path to the Images_png folder
            hu_min:     Lower HU clipping bound (default: -1000, air)
            hu_max:     Upper HU clipping bound (default: 400, soft tissue)
            img_size:   Output image size in pixels (default: 64)
        """
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.img_size = img_size

        # DeepLesion PNG images are stored with a specific intensity mapping:
        # PNG pixel value = (HU + 32768) / 1 (i.e., HU = pixel_value - 32768)
        # This is the standard DeepLesion normalization
        self.HU_OFFSET = 32768

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.images_dir, row['File_name'])
        img = Image.open(img_path)
        img_array = np.array(img, dtype=np.float32)

        # Convert PNG pixel values to Hounsfield Units (HU)
        img_hu = img_array - self.HU_OFFSET

        # Crop to lesion bounding box
        # Bounding box is stored as "x1,y1,x2,y2" string
        bbox = [float(x) for x in str(row['Bounding_boxes']).split(',')]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Add some padding around the lesion
        pad = 10
        h, w = img_hu.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # Crop the lesion region
        cropped = img_hu[y1:y2, x1:x2]

        # Clip Hounsfield Units to meaningful range
        cropped = np.clip(cropped, self.hu_min, self.hu_max)

        # Normalize to [0, 1]
        cropped = (cropped - self.hu_min) / (self.hu_max - self.hu_min)

        # Resize to 64x64
        pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
        pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        final = np.array(pil_img, dtype=np.float32) / 255.0

        # Convert to tensor with shape (1, 64, 64) — 1 channel (grayscale)
        tensor = torch.from_numpy(final).unsqueeze(0)

        # Return the image and the organ label (-1 if unknown)
        label = int(row['Coarse_lesion_type'])
        return tensor, label
```

> **What are Hounsfield Units (HU)?** They are the standard unit for CT scan intensity. Air = -1000 HU, Water = 0 HU, Bone = +400 to +1000 HU, Soft tissue = -100 to +300 HU. We clip to a range relevant to the tissue we care about.

### 5.4 Test the DataLoader

```python
from torch.utils.data import DataLoader
from deeplesion_dataset import DeepLesionDataset

dataset = DeepLesionDataset(
    csv_path="data/stage1_lung_train.csv",
    images_dir="DeepLesion/Images_png"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(loader))

print(f"Batch shape: {images.shape}")   # Should be (4, 1, 64, 64)
print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")  # Should be [0, 1]
print(f"Labels: {labels}")
```

Visualize a few samples to make sure they look right:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(images[i, 0].numpy(), cmap='gray')
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis('off')
plt.savefig("sanity_check_samples.png")
plt.show()
```

### 5.5 Adapt the Training Script

Copy `VAE/RandomSearch_Dirichlet_VAE.py` from the existing repo and modify it:

1. **Replace the data loading section** — swap the LIDC-IDRI data loader with your `DeepLesionDataset`
2. **Update paths** — change `IMAGE_DIR`, `results_path`, `meta_location` to your local paths
3. **Keep everything else the same** — the model architecture, loss function, and training loop should not change for Stage 1

Run Stage 1 training:

```bash
python stage1_lung_only/train_gvae_lung.py
python stage1_lung_only/train_dirvae_lung.py
```

### 5.6 Expected Stage 1 Results

Compare your results to the original paper (Table 1 in arXiv:2311.15719):

| Metric | Target (Keel et al.) | Your Result |
|--------|---------------------|-------------|
| AUC (GVAE) | ~0.85–0.90 | Should be similar |
| AUC (DirVAE) | ~0.95–0.98 | Should be similar |
| Clustering purity | >0.7 | Should be similar |

If your numbers are in this ballpark, Stage 1 is complete and you can move to Stage 2.

---

## 6. Stage 2 — Full Unsupervised Training

**Goal:** Train on all 32,735 patches (no organ labels during training), then probe the latent space with the 9,000 labelled patches.

### 6.1 Prepare the Full Training Set

Create `02_prepare_full_training.py`:

```python
import pandas as pd

df = pd.read_csv("DeepLesion/DL_info.csv")

# Training set: ALL train patches (including unlabelled ones)
train_df = df[df['Train_Val_Test'] == 1].copy()
print(f"Total training patches: {len(train_df)}")  # ~22,921
print(f"Unlabelled: {(train_df['Coarse_lesion_type'] == -1).sum()}")

# Probe set: Val + Test with organ labels
probe_df = df[(df['Train_Val_Test'].isin([2, 3])) &
              (df['Coarse_lesion_type'] != -1)].copy()
print(f"Probe patches with organ labels: {len(probe_df)}")  # ~9,816

# Save
train_df.to_csv("data/stage2_full_train.csv", index=False)
probe_df.to_csv("data/stage2_probe.csv", index=False)

# Check organ distribution in probe set
organ_names = {1: 'Bone', 2: 'Abdomen', 3: 'Mediastinum', 4: 'Liver',
               5: 'Lung', 6: 'Kidney', 7: 'Soft Tissue', 8: 'Pelvis'}
print("\nOrgan distribution in probe set:")
print(probe_df['Coarse_lesion_type'].map(organ_names).value_counts())
```

### 6.2 Train on Full Dataset

The dataset is now 2.5× larger, so training will take longer. The model architecture does **not** need to change — only the data changes.

```bash
# Train GVAE on all 32,735 patches
python stage2_full/train_gvae_full.py

# Train DirVAE on all 32,735 patches
python stage2_full/train_dirvae_full.py
```

> **Tip on training time:** With a GPU, expect 4–12 hours per model depending on hyperparameters. Plan accordingly. You can reduce `num_epochs` or `batch_size` for quick tests.

### 6.3 Extract Latent Vectors for the Probe Set

After training, encode the probe set patches and save their latent vectors. Create `03_extract_latent_vectors.py`:

```python
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from deeplesion_dataset import DeepLesionDataset

# Load your trained model
# (adjust model class and path based on which model you're using)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example for DirVAE:
model = YourDirVAE(latent_dim=32)  # use your actual model class
checkpoint = torch.load("results/dirvae_stage2_best.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Load probe set
probe_dataset = DeepLesionDataset(
    csv_path="data/stage2_probe.csv",
    images_dir="DeepLesion/Images_png"
)
probe_loader = DataLoader(probe_dataset, batch_size=128, shuffle=False)

# Extract latent vectors
all_latents = []
all_labels = []
all_organ_labels = []

with torch.no_grad():
    for images, organ_labels in probe_loader:
        images = images.to(device)

        # For GVAE: encoder returns (mu, logvar) — use mu as the latent vector
        mu, logvar = model.encoder(images)
        latents = mu.cpu().numpy()

        # For DirVAE: encoder returns concentration parameters alpha
        # then sample from Dirichlet
        # alpha = model.encoder(images)
        # latents = alpha.cpu().numpy()

        all_latents.append(latents)
        all_organ_labels.append(organ_labels.numpy())

# Stack into arrays
latent_vectors = np.concatenate(all_latents, axis=0)  # shape: (9816, latent_dim)
organ_labels = np.concatenate(all_organ_labels, axis=0)  # shape: (9816,)

print(f"Latent vectors shape: {latent_vectors.shape}")
print(f"Organ labels shape: {organ_labels.shape}")

# Save for analysis
np.save("results/dirvae_latents_probe.npy", latent_vectors)
np.save("results/probe_organ_labels.npy", organ_labels)
print("Saved latent vectors and labels.")
```

### 6.4 Clustering Analysis

Now measure how well the latent space organizes by organ. Create `04_clustering_analysis.py`:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import classix

# Load latent vectors and labels
latents = np.load("results/dirvae_latents_probe.npy")
organ_labels = np.load("results/probe_organ_labels.npy")

# ── K-Means Clustering ──────────────────────────────────────────────
# Use 8 clusters because there are 8 organ types
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(latents)

# Measure how well clusters align with organ labels
ari_organ = adjusted_rand_score(organ_labels, cluster_assignments)
nmi_organ = normalized_mutual_info_score(organ_labels, cluster_assignments)

print(f"K-Means (8 clusters) vs Organ Labels:")
print(f"  Adjusted Rand Index (ARI): {ari_organ:.4f}")
print(f"  Normalized Mutual Information (NMI): {nmi_organ:.4f}")
# ARI of 0 = random, 1 = perfect. A DirVAE should score significantly higher than GVAE.

# ── Clustering Purity ───────────────────────────────────────────────
def clustering_purity(true_labels, cluster_labels):
    """Purity measures the fraction of correctly assigned points if each
    cluster is assigned to its majority class."""
    total = len(true_labels)
    purity_sum = 0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            continue
        # Find the most common organ label in this cluster
        most_common_count = np.bincount(true_labels[mask].astype(int) + 1).max()
        purity_sum += most_common_count
    return purity_sum / total

purity = clustering_purity(organ_labels, cluster_assignments)
print(f"  Clustering Purity: {purity:.4f}")

# ── CLASSIX Clustering ──────────────────────────────────────────────
clx = classix.CLASSIX(radius=0.5, minPts=5, verbose=0)
clx.fit(latents)
classix_labels = clx.labels_

ari_classix = adjusted_rand_score(organ_labels, classix_labels)
print(f"\nCLASSIX Clustering vs Organ Labels:")
print(f"  ARI: {ari_classix:.4f}")
```

---

## 7. Stage 3 — Dual-Axis Latent Traversals

**Goal:** Find the direction in latent space that corresponds to "organ type" and the direction that corresponds to "malignancy", then move along them independently to generate images.

This is the most impressive result to demonstrate — you visually show that changing one dimension changes the organ while keeping the lesion appearance the same, and vice versa.

### 7.1 Compute Direction Vectors

A **direction vector** is a vector in latent space that points from one class to another. For example, the "malignancy direction" points from the average latent code of benign lesions to the average latent code of malignant lesions.

Create `05_direction_vectors.py`:

```python
import numpy as np

latents = np.load("results/dirvae_latents_probe.npy")
organ_labels = np.load("results/probe_organ_labels.npy")

# ── Organ Direction Vectors ─────────────────────────────────────────
# Compute the mean latent vector for each organ class
organ_means = {}
organ_names = {1: 'Bone', 2: 'Abdomen', 3: 'Mediastinum', 4: 'Liver',
               5: 'Lung', 6: 'Kidney', 7: 'Soft Tissue', 8: 'Pelvis'}

for organ_id, organ_name in organ_names.items():
    mask = organ_labels == organ_id
    if mask.sum() > 0:
        organ_means[organ_id] = latents[mask].mean(axis=0)
        print(f"{organ_name}: {mask.sum()} patches, mean latent computed")

# Overall mean (center of the latent space)
global_mean = latents.mean(axis=0)

# The organ direction from Lung (5) to Liver (4) as an example:
lung_to_liver = organ_means[4] - organ_means[5]
lung_to_liver = lung_to_liver / np.linalg.norm(lung_to_liver)  # normalize to unit vector

# Save all organ means
np.save("results/organ_means.npy", organ_means)
np.save("results/global_mean.npy", global_mean)
print("Direction vectors saved.")
```

> **Note on malignancy:** DeepLesion does not have explicit malignancy labels. For the malignancy direction, you can either:
> (a) Use the LIDC-IDRI lung probe patches (which have malignancy scores from the original dataset)
> (b) Use the RECIST diameter as a proxy (larger lesions tend to be more malignant)
> (c) Train a small MLP classifier on top of the latent vectors using a subset of patches where malignancy info is available, then use the classifier weights as the malignancy direction

### 7.2 Generate Traversal Images

A traversal means: take a starting latent vector, move it step by step along a direction, and decode each step back to an image.

Create `06_latent_traversal.py`:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load model
model = YourDirVAE(latent_dim=32)
checkpoint = torch.load("results/dirvae_stage2_best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load direction vectors
organ_means = np.load("results/organ_means.npy", allow_pickle=True).item()
global_mean = np.load("results/global_mean.npy")

def generate_traversal(model, start_latent, direction, n_steps=10, step_size=0.5):
    """
    Generate a sequence of images by moving in 'direction' from 'start_latent'.

    Args:
        start_latent: Starting point in latent space (numpy array)
        direction:    Direction to move in (unit vector, numpy array)
        n_steps:      Number of images to generate
        step_size:    How far to move each step

    Returns:
        List of PIL images
    """
    images = []
    for i in range(n_steps):
        # Move along the direction
        offset = (i - n_steps // 2) * step_size
        latent = start_latent + offset * direction

        # Decode to image
        latent_tensor = torch.FloatTensor(latent).unsqueeze(0)
        with torch.no_grad():
            reconstructed = model.decoder(latent_tensor)

        # Convert to PIL image
        img_array = reconstructed.squeeze().numpy()
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(img_array))

    return images

# ── Organ Axis Traversal (Lung → Liver) ────────────────────────────
direction = organ_means[4] - organ_means[5]  # Liver - Lung
direction = direction / np.linalg.norm(direction)

# Start from a lung patch's latent code
start_latent = organ_means[5]  # Start at the mean lung latent

organ_images = generate_traversal(model, start_latent, direction, n_steps=10)

# Save as GIF
organ_images[0].save(
    "results/organ_traversal_lung_to_liver.gif",
    save_all=True,
    append_images=organ_images[1:],
    duration=200,  # milliseconds per frame
    loop=0
)
print("Organ traversal GIF saved!")

# ── Plot the traversal as a grid ────────────────────────────────────
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, img in enumerate(organ_images):
    axes[i].imshow(np.array(img), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"Step {i+1}", fontsize=8)
plt.suptitle("Organ Traversal: Lung → Liver")
plt.tight_layout()
plt.savefig("results/organ_traversal_grid.png", dpi=150)
plt.show()
```

---

## 8. Evaluation Metrics

### 8.1 Metrics You Need to Report

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **AUC** | Malignancy classification quality | Train MLP on latents, compute ROC AUC |
| **Clustering Purity** | How organ-pure each cluster is | See clustering analysis section |
| **ARI** | Adjusted Rand Index — cluster/label alignment | `sklearn.metrics.adjusted_rand_score` |
| **NMI** | Normalized Mutual Information | `sklearn.metrics.normalized_mutual_info_score` |
| **MIG Score** | Mutual Information Gap — formal disentanglement metric | See below |

### 8.2 Computing the MIG Score (Important — Novel Contribution)

The **Mutual Information Gap (MIG)** is a formal metric for disentanglement. It measures whether each latent dimension encodes exactly one factor (organ or malignancy) and not the other.

Install the disentanglement library:

```bash
pip install disentanglement-lib  # or implement manually
```

Manual implementation:

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def compute_mig(latent_vectors, labels):
    """
    Compute the Mutual Information Gap (MIG) score.

    Higher MIG → better disentanglement.
    A perfectly disentangled model would have MIG = 1.

    Args:
        latent_vectors: (n_samples, latent_dim) array
        labels:         (n_samples,) array of class labels

    Returns:
        MIG score (float between 0 and 1)
    """
    # Compute mutual information between each latent dimension and the labels
    mi_scores = mutual_info_classif(latent_vectors, labels, discrete_features=False)

    # Normalize by the entropy of the labels
    from scipy.stats import entropy
    label_counts = np.bincount(labels.astype(int) - labels.min())
    label_probs = label_counts / label_counts.sum()
    H_labels = entropy(label_probs)

    normalized_mi = mi_scores / H_labels

    # Sort in descending order
    sorted_mi = np.sort(normalized_mi)[::-1]

    # MIG = difference between the top two mutual information scores
    if len(sorted_mi) >= 2:
        mig = sorted_mi[0] - sorted_mi[1]
    else:
        mig = sorted_mi[0]

    return mig, mi_scores

# Compute MIG for organ axis
mig_organ, mi_per_dim_organ = compute_mig(latent_vectors, organ_labels)
print(f"MIG score (organ axis): {mig_organ:.4f}")

# Visualize which latent dimensions encode organ information
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.bar(range(len(mi_per_dim_organ)), mi_per_dim_organ)
plt.xlabel("Latent Dimension")
plt.ylabel("Mutual Information with Organ Label")
plt.title("Which Latent Dimensions Encode Organ Information?")
plt.savefig("results/mi_per_dimension_organ.png")
plt.show()
```

### 8.3 Results Table Template

Fill this in as you complete experiments:

| Model | Stage | ARI (organ) | NMI (organ) | Purity | MIG (organ) | MIG (mal.) |
|-------|-------|-------------|-------------|--------|-------------|-----------|
| GVAE | 1 (Lung only) | | | | | |
| DirVAE | 1 (Lung only) | | | | | |
| GVAE | 2 (All organs) | | | | | |
| DirVAE | 2 (All organs) | | | | | |

---

## 9. The Key Figure — UMAP Visualization

**UMAP** (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that takes your high-dimensional latent vectors and projects them to 2D for visualization. It preserves the local structure of the data better than PCA.

This is the central figure of your project — a good UMAP plot can show at a glance whether the DirVAE learned meaningful structure.

Create `07_umap_visualization.py`:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap

# Load data
latents = np.load("results/dirvae_latents_probe.npy")
organ_labels = np.load("results/probe_organ_labels.npy")

# ── Fit UMAP ────────────────────────────────────────────────────────
# This may take a few minutes for 9,000 points
print("Fitting UMAP... (this may take a few minutes)")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=30,     # Higher = more global structure preserved
    min_dist=0.1,       # Lower = tighter clusters
    random_state=42
)
embedding = reducer.fit_transform(latents)
print(f"UMAP embedding shape: {embedding.shape}")

# Save embedding for reuse
np.save("results/umap_embedding_dirvae.npy", embedding)

# ── Define Colors and Labels ────────────────────────────────────────
organ_names = {1: 'Bone', 2: 'Abdomen', 3: 'Mediastinum', 4: 'Liver',
               5: 'Lung', 6: 'Kidney', 7: 'Soft Tissue', 8: 'Pelvis'}
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#a65628', '#f781bf', '#999999']

# ── The Key Figure: Two Panels ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Panel (a): Colored by Organ
for organ_id, organ_name in organ_names.items():
    mask = organ_labels == organ_id
    if mask.sum() == 0:
        continue
    ax1.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=colors[organ_id - 1],
        label=organ_name,
        s=5, alpha=0.6
    )
ax1.set_title("(a) DirVAE Latent Space — Coloured by Organ", fontsize=14, fontweight='bold')
ax1.set_xlabel("UMAP Dimension 1")
ax1.set_ylabel("UMAP Dimension 2")
ax1.legend(loc='upper right', markerscale=3, fontsize=9)

# Panel (b): Colored by a proxy for malignancy
# (Use RECIST diameter as proxy — load from probe CSV)
probe_df = pd.read_csv("data/stage2_probe.csv")
# Parse RECIST diameter (largest measurement)
# The column 'Lesion_diameters_Pixel_' contains measurements
# Use the larger diameter as a rough malignancy proxy
diameters = probe_df['RECIST_longest_diameter'].values
size_category = (diameters > diameters.median()).astype(int)  # 0=small, 1=large

sc = ax2.scatter(
    embedding[:, 0], embedding[:, 1],
    c=size_category,
    cmap='RdYlGn_r',  # Red=large/potentially malignant, Green=small/benign
    s=5, alpha=0.6
)
plt.colorbar(sc, ax=ax2, label="Lesion Size (0=small, 1=large)")
ax2.set_title("(b) DirVAE Latent Space — Coloured by Lesion Size (Malignancy Proxy)",
              fontsize=14, fontweight='bold')
ax2.set_xlabel("UMAP Dimension 1")
ax2.set_ylabel("UMAP Dimension 2")

plt.tight_layout()
plt.savefig("results/key_figure_umap_dual.png", dpi=200, bbox_inches='tight')
print("Key figure saved!")
plt.show()
```

### What to Look For in the Plot

**A successful DirVAE result** shows:
- In panel (a): Distinct, separated color clusters — each organ type forms its own region
- In panel (b): A gradient or separation that is **roughly orthogonal** to the organ clusters — indicating malignancy varies independently of organ type

**A failed result** shows:
- Random mixing of colors with no clear clusters
- Panel (b) coloring perfectly matches panel (a) — malignancy and organ are not separated

Both results are valid and publishable. Just analyze and explain what you observe honestly.

---

## 10. Project File Structure

Organize your project like this:

```
DeepLesion_Project/
│
├── DeepLesion/                    # Raw dataset (downloaded)
│   ├── DL_info.csv
│   └── Images_png/
│
├── data/                          # Processed CSV files
│   ├── stage1_lung_train.csv
│   ├── stage1_lung_probe.csv
│   ├── stage2_full_train.csv
│   └── stage2_probe.csv
│
├── src/                           # Reusable code
│   ├── deeplesion_dataset.py      # DataLoader (main new code)
│   ├── models/
│   │   ├── gvae.py                # Gaussian VAE (from original repo)
│   │   └── dirvae.py             # Dirichlet VAE (from original repo)
│   └── metrics.py                 # MIG, purity, ARI functions
│
├── stage1_lung_only/
│   ├── train_gvae_lung.py
│   ├── train_dirvae_lung.py
│   └── evaluate_stage1.ipynb
│
├── stage2_full/
│   ├── train_gvae_full.py
│   └── train_dirvae_full.py
│
├── analysis/
│   ├── 00_data_exploration.ipynb
│   ├── 03_extract_latent_vectors.py
│   ├── 04_clustering_analysis.py
│   ├── 05_direction_vectors.py
│   ├── 06_latent_traversal.py
│   └── 07_umap_visualization.py
│
├── results/
│   ├── *.pt                       # Saved model weights
│   ├── *.npy                      # Latent vectors, labels, embeddings
│   └── *.png / *.gif              # Figures and traversal animations
│
├── requirements.txt
└── README.md
```

---

## 11. Timeline and Division of Work

Suggested timeline for a project due in ~8 weeks:

| Week | Task | Who |
|------|------|-----|
| 1 | Dataset download, environment setup, data exploration | All together |
| 1–2 | Write DeepLesionDataset, verify it loads correctly | 1 person |
| 2 | Adapt training scripts from original repo for DeepLesion | 1 person |
| 2–3 | **Stage 1:** Train GVAE + DirVAE on lung-only patches | 1–2 people |
| 3 | Evaluate Stage 1, compare to original paper | 1 person |
| 4–5 | **Stage 2:** Train GVAE + DirVAE on full dataset (all 32k patches) | All GPUs needed |
| 5 | Extract latent vectors, run clustering analysis | 1 person |
| 5–6 | UMAP visualizations, MIG score computation | 1 person |
| 6 | **Stage 3:** Direction vectors and latent traversal GIFs | 1–2 people |
| 7 | Write analysis, compare GVAE vs DirVAE results | All |
| 8 | Report writing, figures, presentation prep | All |

### Suggested Task Split

- **Camille:** DataLoader + preprocessing pipeline, Stage 1 training
- **Franklin:** Stage 2 full training (needs GPU access), model checkpointing
- **Loïc:** Evaluation metrics (ARI, NMI, MIG), clustering analysis notebooks
- **Flavio:** UMAP visualization, latent traversal code, key figures

---

## 12. Glossary

| Term | Simple Explanation |
|------|-------------------|
| **VAE** | Neural network that compresses images to small codes and reconstructs them |
| **GVAE** | Standard VAE where codes follow a Normal distribution |
| **DirVAE** | VAE where codes follow a Dirichlet distribution (encourages independence of dimensions) |
| **Latent vector** | The compressed code — a small set of numbers that represent an image |
| **Latent space** | The space of all possible latent vectors |
| **Disentanglement** | When each latent dimension encodes a different independent feature |
| **Probe set** | Labelled data used AFTER training to evaluate what was learned — never seen during training |
| **HU (Hounsfield Units)** | Standard unit for CT scan pixel intensity |
| **UMAP** | Algorithm to visualize high-dimensional data in 2D |
| **ARI** | Adjusted Rand Index — measures similarity between two clusterings (0 = random, 1 = perfect) |
| **NMI** | Normalized Mutual Information — how much cluster assignments reveal about true labels |
| **Purity** | Fraction of points in each cluster that belong to the majority class |
| **MIG** | Mutual Information Gap — formal metric for disentanglement quality |
| **KL Divergence** | Part of the VAE loss; measures how far the learned distribution is from the prior |
| **Beta-annealing** | Gradually increasing the KL term weight during training for better learning |
| **SSIM/MS-SSIM** | Structural similarity metrics — measure image quality beyond pixel-level error |
| **RECIST diameter** | Standard clinical measurement of lesion size in CT scans |
| **Latent traversal** | Moving along a direction in latent space step-by-step and decoding each step to an image |

---

## Quick Start Checklist

Use this to track your progress:

- [ ] Python environment set up and all packages installed
- [ ] DeepLesion dataset downloaded and extracted
- [ ] `DL_info.csv` loaded and explored in a notebook
- [ ] `DeepLesionDataset` written and tested (images load and look correct)
- [ ] Stage 1 training scripts adapted from original repo
- [ ] Stage 1 GVAE trained and evaluated
- [ ] Stage 1 DirVAE trained and evaluated
- [ ] Stage 1 results match original paper (AUC ~0.95+)
- [ ] Stage 2 full training data prepared (22,921 patches)
- [ ] Stage 2 GVAE trained
- [ ] Stage 2 DirVAE trained
- [ ] Latent vectors extracted for probe set (9,816 patches)
- [ ] Clustering analysis run (K-Means + CLASSIX)
- [ ] MIG score computed for both GVAE and DirVAE
- [ ] UMAP plots generated (both panels)
- [ ] Direction vectors computed (organ + malignancy proxy)
- [ ] Latent traversal GIFs generated
- [ ] Results table filled in for all experiments
- [ ] Report written and figures finalized
