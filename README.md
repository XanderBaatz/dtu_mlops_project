# dtu_mlops_project

DTU machine learning operations project

## Project description
The goal of this project is to implement, optimize, and evaluate a steerable group equivariant convolutional neural network (GCNN) using escnn, a PyTorch-based library that enables the construction of convolutional networks with  equivariance to symmetry groups such as rotations and reflections (and even reflections). The project focuses on applying these models to a realworld medical imaging task in histopathology, where robustness to these geometric transformations is important.

Standard convolutional neural networks are equivariant only to translations. However, histopathological images often contain additional symmetries, becuase  tissue samples can appear in arbitrary orientation. Rotations and reflections of a tissue patch do not change its underlying cancer status, yet standard CNNs must learn these symmetries through data augmentation. By explicitly encoding symmetry through group equivariance, steerable GCNNs offer a inbuilt way to improve generalization by implement equviarant filters using fourier coefficients.

The primary dataset used in this project is PatchCamelyon (PCam), a  dataset for detecting metastatic cancer in lymph node histology images. The dataset consists of 96×96 RGB image patches, each labeled according to whether it contains tumor tissue, resulting in a binary classification problem. PCam is well suited for this study because the class label is invariant under rotations and reflections, making it an ideal testbed for equivariant models. The dataset will initially be used with its standard training, validation, and test splits.

Two main models will be implemented and compared. The first is a baseline CNN, consisting of standard convolutional layers, nonlinearities, and pooling operations, serving as a reference for performance and training behavior (and maybe even with data augmentation). The second is a steerable GCNN, implemented using escnn, which replaces standard convolutions with equivariant convolutions defined over different symmetry groups. Multiple symmetry configurations will be explored to study their impact on performance and robustness.

Both model types will be trained and optimized using a systematic hyperparameter study, including learning rate, depth, width, and regularization. Final evaluation will be performed using the best-performing hyperparameter configurations.

## Project structure

The directory structure of the project looks like this:

```txt
├── .devcontainer/            # VS Code dev container configuration
├── .github/                  # Github actions and workflows
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml        # Automated testing pipeline
├── .dvcignore                # DVC ignore patterns
├── .gcloudignore             # Google Cloud ignore patterns (for Cloud Build)
├── configs/                  # Hydra configuration files
│   ├── train.yaml            # Main training config
│   ├── config.yaml           # Base config
│   ├── data/                 # Data configuration
│   │   └── fashion_mnist.yaml
│   ├── model/                # Model configurations
│   │   ├── cnn.yaml
│   │   └── c8.yaml
│   ├── trainer/              # PyTorch Lightning trainer configs
│   │   ├── cpu.yaml
│   │   ├── gpu.yaml
│   │   └── mps.yaml
│   ├── callbacks/            # Lightning callback configs
│   ├── logger/               # Logger configurations (CSV, WandB)
│   ├── paths/                # Path configurations
│   ├── hydra/                # Hydra framework configs
│   ├── debug/                # Debug settings
│   ├── hparams_search/       # Hyperparameter search configs
│   ├── extras/               # Extra configurations
│   └── vertex_ai/            # Google Vertex AI job configs
│       └── config_cpu.yaml
├── data/                     # Data directory
│   └── FashionMNIST/         # Fashion MNIST dataset
├── dockerfiles/              # Docker configurations
│   ├── api.dockerfile        # FastAPI server Dockerfile
│   ├── train.dockerfile      # Training Dockerfile
│   └── cloud.dockerfile      # Cloud training Dockerfile
├── docs/                     # Project documentation
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── gcp/                      # Google Cloud Platform configs
│   ├── cloudbuild.yaml       # Cloud Build configuration
│   └── policy.yaml           # IAM policies
├── models/                   # Trained model checkpoints
│   └── model.pth
├── notebooks/                # Jupyter notebooks for exploration
├── profiler/                 # Performance profiling outputs
├── reports/                  # Analysis reports and figures
│   └── figures/
│       ├── sample_images.png
│       └── train_label_distribution.png
├── src/                      # Source code
│   └── dtu_mlops_project/
│       ├── __init__.py
│       ├── api.py            # FastAPI REST API
│       ├── apifile.py        # API file utilities
│       ├── data.py           # Data loading and preprocessing
│       ├── evaluate.py       # Model evaluation
│       ├── model.py          # Model architecture definitions
│       ├── setup_check.py    # Environment setup validation
│       ├── train.py          # Training script with Hydra config
│       └── visualize.py      # Visualization utilities
├── tests/                    # Unit and integration tests
│   ├── __init__.py
│   ├── test_api.py           # API tests
│   ├── test_data.py          # Data pipeline tests
│   └── test_model.py         # Model tests
├── .gitignore
├── .pre-commit-config.yaml   # Pre-commit hooks (linting, formatting)
├── AGENTS.md                 # Instructions for autonomous coding agents
├── cloudbuild.yaml           # Cloud Build pipeline for Docker image
├── data.dvc                  # DVC data versioning file
├── LICENSE
├── pyproject.toml            # Python project configuration (dependencies, metadata)
├── README.md                 # This file
├── tasks.py                  # Invoke task definitions (uv run invoke --list)
└── uv.lock                   # Locked dependency versions (uv package manager)
```

## Component Descriptions

### `src/dtu_mlops_project/`
- **train.py**: Main training script using PyTorch Lightning and Hydra for config management
- **data.py**: Data loading, preprocessing, and augmentation (RotatedFashionMNIST dataset)
- **model.py**: CNN and GCNN model architectures
- **evaluate.py**: Model evaluation and metrics computation
- **api.py**: FastAPI REST API for model inference
- **visualize.py**: Training visualization and analysis utilities

### `configs/`
Hierarchical Hydra configuration system for reproducible experiments:
- Override configs with: `python train.py data=fashion_mnist trainer=gpu model=cnn`
- Supports experiment configs, hyperparameter sweeps, and cloud deployment configs

### `dockerfiles/`
- **train.dockerfile**: For local training with GPU/CPU support
- **cloud.dockerfile**: Optimized for Vertex AI custom training jobs
- **api.dockerfile**: FastAPI server for model serving

### Cloud Integration
- **cloudbuild.yaml**: Automated Docker image building and pushing to Artifact Registry
- **configs/vertex_ai/config_cpu.yaml**: Vertex AI job configuration (hyperparameters, data paths)
- **data.dvc**: DVC integration for versioned data in GCS bucket




Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Quick Start

### Installation
```sh
# Install dependencies using uv package manager
uv sync

# Verify setup
uv run invoke setup-check
```

### Local Training
```sh
# Train with default config (CPU, Fashion MNIST)
uv run python src/dtu_mlops_project/train.py

# Train with custom config
uv run python src/dtu_mlops_project/train.py trainer=gpu model=cnn data.batch_size=64

# Train with experiment config
uv run python src/dtu_mlops_project/train.py experiment=best_model
```

### Hyperparameter Sweeps
```sh
# WandB sweep
uv run python src/dtu_mlops_project/train.py --multirun hparams_search=wandb_sweep
```

### Cloud Training (Vertex AI)
```sh
# Build and push Docker image to Artifact Registry
gcloud builds submit . --config cloudbuild.yaml --substitutions=_IMAGE=api,_TAG=$(git rev-parse --short HEAD)

# Submit Vertex AI training job
gcloud ai custom-jobs create --region=europe-west1 --display-name=training-job --config=configs/vertex_ai/config_cpu.yaml

# Stream job logs
gcloud ai custom-jobs stream-logs projects/YOUR_PROJECT/locations/europe-west1/customJobs/JOB_ID
```

### Testing
```sh
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src
```

### Code Quality
```sh
# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Available Commands
```sh
# See all Invoke tasks
uv run invoke --list

# Common tasks
uv run invoke setup-check      # Verify environment setup
uv run invoke train            # Run training
uv run invoke evaluate         # Evaluate model
uv run invoke test             # Run tests
```
With hyperparameters and epochs:
```sh
uv run python src/dtu_mlops_project/train.py model=c8 trainer.max_epochs=1
```
## Credits
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- [Machine Learning Operations](https://github.com/SkafteNicki/dtu_mlops)
