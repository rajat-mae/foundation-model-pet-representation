# Patient-Level DINOv2 PET Representation Learning

A research notebook for extracting and adapting patient-level PET representations from 3D NIfTI volumes using a pretrained DINOv2 Vision Transformer.

The workflow converts each PET volume into a bag of neighbouring axial slice triplets, encodes the triplets with DINOv2, aggregates slice representations with attention-based patient pooling, and supports several representation-learning branches:

| Branch | Notebook name | Purpose |
|---|---|---|
| A | `frozen_dino_baseline` | Extract patient embeddings using a frozen DINOv2 backbone. |
| B | `supervised_frozen_classifier` | Train a supervised patient classifier, with optional selective tuning of late DINOv2 layers. |
| C | `dino_ssl` | Train a patient-level DINO-style student–teacher self-supervised model. |
| D | `dino_jepa_slice` | Predict latent targets from spatially masked PET slice triplets. |
| E | `dino_jepa_cross_slice` | Predict a held-out slice representation from neighbouring slice context. |

The notebook also exports patient embeddings, checkpoints, training diagnostics, attention or embedding-conditioned region maps, PCA projections, clustering-related outputs, and centroid-distance analyses.

> **Research use only.** This code is not a medical device, has not been clinically validated, and must not be used for diagnosis or patient-management decisions.

---

## 1. Repository contents

A minimal standalone repository should contain:

```text
patient-dinov2-pet-representations/
├── Foundation_model_PET_representation.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/                              # local only; never commit patient data
│   ├── train/
│   ├── val/
│   └── oac/
├── outputs/                           # generated locally
├── pretrained_dinov2_weights/         # downloaded locally
└── third_party/
    └── dinov2/                        # cloned official DINOv2 source
```

The supplied `.gitignore` excludes data, model weights, checkpoints, generated outputs, virtual environments, and the local DINOv2 checkout.

---

## 2. Important execution notes

Read these before running the notebook from top to bottom.

1. **Linux or WSL2 is recommended.** The official DINOv2 training/evaluation environment is Linux-oriented. Native Windows can work for portions of this notebook, but WSL2 with an NVIDIA GPU is the more reliable Windows setup.
2. **A CUDA GPU is strongly recommended.** CPU execution is suitable for a structural smoke test or very small inference run, but full ViT-B patient-level SSL or JEPA training will be extremely slow.
3. **Edit the `Config` cell before loading data or a model.** The notebook currently contains example WSL paths.
4. **Do not run both JEPA implementations in one experiment.** The notebook contains an original DINO-JEPA cell and a later improved DINO-JEPA cell. Both use overlapping branch and checkpoint names. Run one implementation only—preferably the later cell titled **“Improved Branch D/E”**—or use different output directories.
5. **Branch-C export cells require DINO-SSL checkpoints.** If `dino_ssl` is not enabled, skip the cells that export and compare `ssl_best.pt` and `ssl_last.pt`.
6. **The final 3D-PCA and external-cohort cells are standalone analyses.** They reset `ROOT` to a hard-coded `/mnt/d/...` path. Update those paths before running them, or skip those cells.
7. **Patient identifiers are derived from filenames.** Use de-identified filenames and do not commit imaging data, metadata, notebook outputs containing identifiers, or local filesystem paths.

---

## 3. System requirements

### Recommended

- Ubuntu 22.04/24.04 or Windows 10/11 with WSL2 Ubuntu
- Python 3.9–3.11
- NVIDIA GPU with CUDA support
- At least 16 GB system RAM
- Substantial free disk space for PET data, DINOv2 weights, checkpoints, and embeddings

The default model is DINOv2 ViT-B/14 with four register tokens. Full multi-branch training may require considerably more GPU memory and time than the frozen baseline.

### CPU-only systems

The notebook automatically falls back to CPU because its device cell uses:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For a CPU smoke test, use the mock backbone and very small settings described later. Do not expect practical full-training runtimes on a standard CPU.

### Apple Silicon

The notebook does not currently select Apple `mps`. An experimental device selection is:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Some DINOv2 or mixed-precision operations may still require adjustment on MPS. CPU fallback is safer when an operation is unsupported.

---

## 4. Clone this repository



```bash
git clone https://github.com/rajat-mae/foundation-model-pet-representation.git
cd <YOUR-REPOSITORY>
```

Replace the placeholders with the actual GitHub account and repository name.

---

## 5. Create a Python environment

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### WSL2, Linux, or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Conda may be used instead:

```bash
conda create -n pet-dinov2 python=3.10 -y
conda activate pet-dinov2
```

---

## 6. Install PyTorch

Install PyTorch and torchvision **before** the remaining packages. The correct command depends on the operating system, GPU, driver, and supported CUDA build.

Use the official selector:

- [PyTorch — Start Locally](https://pytorch.org/get-started/locally/)

Example CPU installation:

```bash
pip install torch torchvision
```

For an NVIDIA GPU, use the CUDA-specific command generated by the PyTorch selector rather than blindly copying a CUDA command from another computer.

Verify the installation:

```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 7. Install notebook dependencies

```bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name pet-dinov2 --display-name "PET DINOv2"
```

Start Jupyter:

```bash
jupyter lab
```

Open `Foundation_model_PET_representation.ipynb` and select the **PET DINOv2** kernel.

---

## 8. Install the official DINOv2 source

The notebook’s recommended loading mode constructs the Vision Transformer from the official DINOv2 source and then loads an official checkpoint manually.

From the repository root:

```bash
mkdir -p third_party
git clone --depth 1 https://github.com/facebookresearch/dinov2.git third_party/dinov2
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force third_party
git clone --depth 1 https://github.com/facebookresearch/dinov2.git third_party/dinov2
```

The notebook adds the configured DINOv2 directory to `sys.path`, so an editable package installation is not required for its default manual-loading path. If imports fail, install the dependencies supplied by the official DINOv2 repository:

```bash
pip install -r third_party/dinov2/requirements.txt
```

Be aware that the official full training stack uses version constraints that may differ from a newer local PyTorch installation. For this notebook, use a consistent torch/torchvision pair and test the frozen baseline before attempting SSL training.

Official source:

- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)

---

## 9. DINOv2 pretrained weights

The notebook supports automatic download or a manually supplied checkpoint. By default it uses:

```python
CFG.backbone_name = "dinov2_vitb14_reg"
```

This is DINOv2 ViT-B/14 with four register tokens and a 768-dimensional embedding.

The checkpoint map embedded in the notebook points to Meta-hosted files:

| Backbone | Embedding dimension | Official checkpoint |
|---|---:|---|
| `dinov2_vits14` | 384 | [ViT-S/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |
| `dinov2_vitb14` | 768 | [ViT-B/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) |
| `dinov2_vitl14` | 1024 | [ViT-L/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) |
| `dinov2_vitg14` | 1536 | [ViT-g/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth) |
| `dinov2_vits14_reg` | 384 | [ViT-S/14 + registers](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth) |
| `dinov2_vitb14_reg` | 768 | [ViT-B/14 + registers](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) |
| `dinov2_vitl14_reg` | 1024 | [ViT-L/14 + registers](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) |
| `dinov2_vitg14_reg` | 1536 | [ViT-g/14 + registers](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth) |

### Automatic download

Use:

```python
CFG.backbone_source = "manual_local_weights"
CFG.manual_checkpoint_path = None
CFG.auto_download_checkpoint = True
CFG.checkpoint_cache_dir = str(Path.cwd() / "pretrained_dinov2_weights")
```

The notebook downloads the selected file when it is absent.

### Use a pre-downloaded checkpoint

```python
CFG.backbone_source = "manual_local_weights"
CFG.manual_checkpoint_path = str(
    Path.cwd()
    / "pretrained_dinov2_weights"
    / "dinov2_vitb14_reg4_pretrain.pth"
)
CFG.auto_download_checkpoint = False
```

### Remote PyTorch Hub alternative

```python
CFG.backbone_source = "remote_hub"
CFG.backbone_name = "dinov2_vitb14_reg"
```

This is simpler but requires internet access when the model is not already cached.

### Structural mock model

```python
CFG.backbone_source = "mock_smoke"
```

The mock model is only for checking data loading, tensor shapes, branch wiring, and output creation. Its embeddings are not meaningful pretrained DINOv2 representations.

---

## 10. Prepare the PET dataset

The notebook reads three-dimensional NIfTI files ending in `.nii` or `.nii.gz`.

### Labeled training and validation cohorts

Each immediate subdirectory under `train` or `val` is treated as a class. Every NIfTI file is treated as one patient.

```text
data/
├── train/
│   ├── class_1/
│   │   ├── patient_001.nii.gz
│   │   └── patient_002.nii.gz
│   ├── class_2/
│   │   ├── patient_101.nii.gz
│   │   └── patient_102.nii.gz
│   └── class_3/
│       └── patient_201.nii.gz
├── val/
│   ├── class_1/
│   │   └── patient_301.nii.gz
│   ├── class_2/
│   │   └── patient_401.nii.gz
│   └── class_3/
│       └── patient_501.nii.gz
└── oac/
    ├── external_001.nii.gz
    └── external_002.nii.gz
```

### Unlabeled external/OAC cohort

Files under `oac` are recursively discovered without class labels:

```text
data/oac/**/*.nii.gz
```

### Data assumptions

- Each file must contain one finite 3D volume.
- Axial slices are assumed along array axis `2`.
- Patient IDs are generated from the NIfTI filename after removing `.nii` or `.nii.gz`.
- Filenames should be unique and de-identified.
- Train and validation class-folder names should match.
- Confirm image orientation and preprocessing before scientific use. The notebook does not reorient images to a canonical anatomical orientation.
- The notebook does not perform lesion segmentation, PET SUV conversion, scanner harmonisation, or spatial resampling by default.

---

## 11. Make the configuration portable

In the `Config` cell, replace the example `/mnt/...` paths with paths relative to the repository:

```python
data_root: str = str(Path.cwd() / "data")
output_root: str = str(Path.cwd() / "outputs")

backbone_source: str = "manual_local_weights"
dinov2_repo_dir: str = str(Path.cwd() / "third_party" / "dinov2")
checkpoint_cache_dir: str = str(Path.cwd() / "pretrained_dinov2_weights")
manual_checkpoint_path: Optional[str] = None
auto_download_checkpoint: bool = True
```

On native Windows, start with:

```python
num_workers: int = 0
```

Once the notebook runs reliably, increase workers if required.

The input directories resolve to:

```text
<data_root>/train
<data_root>/val
<data_root>/oac
```

Generated files resolve under:

```text
<output_root>/checkpoints
<output_root>/artifacts
<output_root>/plots
```

---

## 12. Select an experiment branch

Set `CFG.branches_to_run` before executing the branch-controller cell.

### Frozen baseline only

This is the recommended first real run:

```python
branches_to_run: Tuple[str, ...] = (
    "frozen_dino_baseline",
)
```

### Frozen baseline plus supervised classifier

```python
branches_to_run: Tuple[str, ...] = (
    "frozen_dino_baseline",
    "supervised_frozen_classifier",
)
```

### DINO-style self-supervised training

```python
branches_to_run: Tuple[str, ...] = (
    "dino_ssl",
)
```

### Improved cross-slice DINO-JEPA

```python
branches_to_run: Tuple[str, ...] = (
    "dino_jepa_cross_slice",
)
```

When running a JEPA branch, execute only one of the notebook’s JEPA training implementations. The later **Improved Branch D/E** cell is the recommended version.

### All conceptual branches

```python
branches_to_run: Tuple[str, ...] = (
    "frozen_dino_baseline",
    "supervised_frozen_classifier",
    "dino_ssl",
    "dino_jepa_slice",
    "dino_jepa_cross_slice",
)
```

Do not begin with all branches. Validate each branch independently and use a separate `output_root` for each experiment.

---

## 13. First smoke test

Use a small de-identified subset and conservative settings:

```python
CFG.backbone_source = "mock_smoke"
CFG.branches_to_run = ("frozen_dino_baseline",)

CFG.base_resize = 224
CFG.bag_size_train = 2
CFG.bag_size_eval = 2
CFG.n_global_crops = 2
CFG.n_local_crops = 0

CFG.train_batch_size = 1
CFG.eval_batch_size = 1
CFG.num_workers = 0
CFG.epochs = 1
CFG.use_amp = False
```

Run the notebook through:

1. imports and environment check;
2. configuration;
3. patient discovery;
4. dataset and DataLoader creation;
5. first-batch diagnostics;
6. mock-backbone construction;
7. frozen-baseline embedding export.

Check that:

- train/validation/OAC patient counts are correct;
- first-batch tensors are finite;
- tensor shapes are as expected;
- embeddings and metadata are written under `outputs/artifacts/branches/frozen_dino_baseline/`.

The mock smoke test verifies execution only. Repeat with an official pretrained backbone before interpreting any embeddings.

---

## 14. Recommended staged workflow

### Stage 1 — Data validation

Run discovery and visual sanity-check cells.

Confirm:

- class counts;
- image shapes;
- slice orientation;
- finite intensity values;
- absence of duplicated patients across train and validation;
- de-identification.

### Stage 2 — Frozen DINOv2 baseline

Use:

```python
CFG.backbone_source = "manual_local_weights"
CFG.backbone_name = "dinov2_vitb14_reg"
CFG.branches_to_run = ("frozen_dino_baseline",)
```

This establishes a no-adaptation reference.

### Stage 3 — Supervised comparator

Enable:

```python
CFG.branches_to_run = ("supervised_frozen_classifier",)
```

Begin with the backbone frozen or only the final transformer block unfrozen at a much smaller learning rate than the classifier head.

### Stage 4 — DINO-SSL

Enable only:

```python
CFG.branches_to_run = ("dino_ssl",)
```

Monitor:

- training and validation losses;
- teacher and student entropy;
- teacher output standard deviation;
- gradient norms;
- learning rate;
- DINO centre norm;
- CUDA memory.

### Stage 5 — JEPA ablation

Run either the slice-masking or cross-slice branch in a new output directory. Do not overwrite checkpoints from another JEPA implementation.

### Stage 6 — Representation evaluation

Compare branches using:

- validation classification performance;
- PCA only as a visualisation, not as proof of separation;
- clustering metrics;
- retrieval or nearest-neighbour performance;
- class-centroid distances;
- external-cohort projections;
- bootstrap confidence intervals where appropriate.

Fit preprocessing, PCA, clustering, and classifiers on training data only when estimating out-of-sample performance.

---

## 15. PET preprocessing implemented by the notebook

For each patient:

1. Load one 3D NIfTI volume with nibabel.
2. Apply percentile clipping and scale intensities to unsigned 8-bit values.
3. Choose a fixed-size set of axial slice centres.
4. For each centre slice, stack the previous, centre, and next slices as a three-channel image.
5. Resize and pad images to a multiple of DINOv2’s patch size, 14.
6. Apply augmentations for SSL training.
7. Encode each triplet with DINOv2.
8. Aggregate slice tokens into one patient token with learned attention pooling.

Two intensity modes are available:

```python
CFG.normalization_scope = "slice"
```

or:

```python
CFG.normalization_scope = "patient"
```

These modes are scientifically different and should be treated as a preprocessing ablation rather than interchangeable implementation details.

---

## 16. Main outputs

### Branch-specific embeddings

```text
outputs/artifacts/branches/<branch_name>/
├── train_embeddings.npy
├── val_embeddings.npy
├── oac_embeddings.npy
├── train_embeddings_metadata.csv
├── val_embeddings_metadata.csv
├── oac_embeddings_metadata.csv
├── train_patient_ids.npy
├── val_patient_ids.npy
├── oac_patient_ids.npy
├── train_labels.npy
├── val_labels.npy
├── oac_labels.npy
└── manifest.json
```

### DINO-SSL checkpoints

```text
outputs/checkpoints/
├── ssl_best.pt
└── ssl_last.pt
```

### Supervised branch

Typical outputs include:

```text
outputs/checkpoints/
├── supervised_frozen_classifier_best.pt
└── supervised_frozen_classifier_last.pt

outputs/artifacts/branches/supervised_frozen_classifier/
├── supervised_training_history.csv
├── train_predictions.csv
├── val_predictions.csv
└── ...
```

### JEPA branches

Typical checkpoints include:

```text
outputs/checkpoints/
├── dino_jepa_slice_best.pt
├── dino_jepa_slice_last.pt
├── dino_jepa_cross_slice_best.pt
└── dino_jepa_cross_slice_last.pt
```

### Training diagnostics

```text
outputs/artifacts/training_history.csv

outputs/plots/
├── ssl_loss_curve.png
├── ssl_optimisation_diagnostics.png
├── ssl_entropy_diagnostics.png
└── ssl_center_output_std_diagnostics.png
```

### Interpretability and post-processing

The notebook can additionally generate:

- slice attention overlays;
- patch-token similarity maps;
- embedding-occlusion maps;
- best-versus-last checkpoint PCA plots;
- branch-wise 3D-PCA panels;
- class-centroid distance matrices;
- external-cohort centroid comparisons.

Interpretability maps show model sensitivity or token alignment. They do not establish pathological localisation or causal biological relevance.

---

## 17. Resume or reuse trained models

The notebook saves state dictionaries and configuration metadata inside `.pt` checkpoints.

Before loading a checkpoint, reconstruct the same:

- DINOv2 backbone;
- register-token configuration;
- patient pooling module;
- projection or prediction head;
- embedding dimension;
- branch architecture.

Then load with:

```python
payload = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(payload["teacher"], strict=True)
```

The exact payload key differs by branch. Inspect:

```python
payload.keys()
```

before loading an unfamiliar checkpoint.

Never load untrusted PyTorch checkpoint files. Pickled checkpoints can execute code during deserialisation.

---

## 18. Common problems

### `fatal: not a git repository`

Initialise the local folder first:

```bash
git init -b main
```

### DINOv2 constructors cannot be imported

Confirm that:

```python
CFG.dinov2_repo_dir = str(Path.cwd() / "third_party" / "dinov2")
```

and that this file exists:

```text
third_party/dinov2/dinov2/models/vision_transformer.py
```

### `Cannot find callable dinov2_vitb14_reg in hubconf`

Use the notebook’s recommended manual loader:

```python
CFG.backbone_source = "manual_local_weights"
```

and point `CFG.dinov2_repo_dir` to a current official DINOv2 checkout.

### Checkpoint download fails

Download the selected official checkpoint manually, place it under `pretrained_dinov2_weights/`, and set `CFG.manual_checkpoint_path`.

### CUDA out of memory

Reduce, in this order:

```python
CFG.base_resize
CFG.bag_size_train
CFG.bag_size_eval
CFG.frozen_backbone_chunk_size
CFG.eval_chunk_size
```

Also:

- keep `train_batch_size = 1`;
- use gradient accumulation;
- keep `n_local_crops = 0` until stable;
- start with a frozen backbone;
- use ViT-S instead of ViT-B;
- run one branch per process;
- restart the kernel between large experiments.

### DataLoader hangs on Windows

Use:

```python
CFG.num_workers = 0
```

### No training or validation patients discovered

Check the expected folder structure and extensions. Labeled files must be inside class subdirectories under `train` and `val`.

### Branch-C export fails when DINO-SSL was not run

Skip the best/last DINO-SSL export and PCA cells, or first generate:

```text
outputs/checkpoints/ssl_best.pt
outputs/checkpoints/ssl_last.pt
```

### JEPA runs twice

The notebook contains original and improved JEPA implementations. Disable one of them or assign different branch/checkpoint directories.

### Final PCA cells cannot find files

Update the hard-coded `ROOT` variable in the standalone analysis cells. Those cells do not automatically reuse `CFG.output_root`.

---

## 19. Reproducibility

The notebook seeds Python, NumPy, and PyTorch:

```python
SEED = 1024
```

For stricter reproducibility, add:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This can reduce speed and does not guarantee bitwise-identical results across all GPU models, CUDA versions, or operations.

Record for every experiment:

- Git commit hash;
- Python version;
- PyTorch and torchvision versions;
- CUDA and GPU model;
- DINOv2 repository commit;
- checkpoint filename and checksum;
- complete `CFG`;
- patient-level train/validation split;
- image preprocessing;
- branch and executed cell version.

The notebook manifests and checkpoint configuration fields should be retained with reported results.

---

## 20. Data privacy and repository hygiene

Do not commit:

- PET, CT, DICOM, NIfTI, or derived patient images;
- identifiable filenames or metadata;
- clinical spreadsheets;
- local absolute paths;
- notebook outputs containing patient identifiers;
- DINOv2 weights;
- trained checkpoints;
- exported embeddings;
- access tokens or credentials.

Before making the repository public, clear notebook outputs:

```bash
jupyter nbconvert \
  --ClearOutputPreprocessor.enabled=True \
  --inplace Foundation_model_PET_representation.ipynb
```

Then inspect:

```bash
git status
git diff --cached
```

For clinical research, follow the applicable ethics approval, consent or waiver, data-sharing agreement, institutional policy, and privacy legislation.

---

## 21. Create a new GitHub repository

### Option A — GitHub CLI

Authenticate once:

```bash
gh auth login
```

From the repository folder:

```bash
git init -b main
git add README.md requirements.txt .gitignore Foundation_model_PET_representation.ipynb
git status
git commit -m "Initial release: patient-level DINOv2 PET representations"
```

Create and push a public repository:

```bash
gh repo create patient-dinov2-pet-representations \
  --public \
  --description "Patient-level PET representation learning with DINOv2, DINO-SSL and DINO-JEPA" \
  --source=. \
  --remote=origin \
  --push
```

Use `--private` instead of `--public` when appropriate.

### Option B — GitHub website

1. Create an empty repository on GitHub.
2. Do not initialise it with another README, `.gitignore`, or licence.
3. From the local repository folder, run:

```bash
git init -b main
git add README.md requirements.txt .gitignore Foundation_model_PET_representation.ipynb
git commit -m "Initial release: patient-level DINOv2 PET representations"
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote -v
git remote set-url origin https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
git push -u origin main
```

For future changes:

```bash
git add README.md Foundation_model_PET_representation.ipynb
git commit -m "Update PET representation workflow"
git push
```

---

## 22. Suggested repository description

> Patient-level PET representation learning from 3D NIfTI volumes using pretrained DINOv2, attention-based slice pooling, DINO-style self-distillation, supervised adaptation, and JEPA-style latent prediction.

Suggested topics:

```text
medical-imaging
pet
nuclear-medicine
dinov2
self-supervised-learning
vision-transformer
representation-learning
jepa
pytorch
nifti
```

---

## 23. Citation and attribution

This repository builds on DINOv2 and the register-token extension. Cite the relevant original publications when using the pretrained models or method in research:

- Oquab M, et al. **DINOv2: Learning Robust Visual Features without Supervision.**
- Darcet T, et al. **Vision Transformers Need Registers.**

Also cite or acknowledge the software versions and any additional methods used in the patient-level adaptation.

This repository should not redistribute Meta’s pretrained weights. Users should retrieve them from the official DINOv2 sources and comply with the applicable code and model licences.

---

## 24. Scientific limitations

- DINOv2 was pretrained on natural images, not PET.
- Three-channel neighbouring-slice triplets are a 2.5D representation, not a native 3D transformer input.
- Per-slice normalisation can remove absolute uptake relationships.
- Patient-level attention pooling does not provide spatially registered pathology correspondence.
- PCA separation is descriptive and does not demonstrate clinical validity.
- Centroid distances are sensitive to cohort composition, preprocessing, class imbalance, and dimensionality.
- The OAC/external branch is unlabeled in the main loader, so evaluation requires independently curated labels.
- External validation, scanner/site robustness, calibration, uncertainty, and clinically meaningful endpoints remain necessary.

---

## 25. Licence

No project licence is assigned by this README. Add a repository licence only after confirming ownership, institutional requirements, third-party obligations, and the intended research or commercial use.

The DINOv2 source code and pretrained model artefacts remain governed by their own official licence terms.
