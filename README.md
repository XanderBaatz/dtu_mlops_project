# dtu_mlops_project

DTU machine learning operations project

## Project description


The goal of this project is to implement, optimize, and evaluate a steerable group equivariant convolutional neural network (GCNN) using escnn, a PyTorch-based library that enables the construction of convolutional networks with equivariance to symmetry groups such as rotations and reflections. The project focuses on applying these models to an image classification task in computer vision, where robustness to geometric transformations is important.

Standard convolutional neural networks are equivariant only to translations. However, many image classification tasks consist of additional symmetries, because objects can appear in arbitrary orientations within the image. Rotations and reflections of an object do not change its underlying class, but standard CNNs must learn these symmetries through data augmentation or pooling methods. By explicitly encoding symmetry through group equivariance kernels, steerable GCNNs offer an inbuilt way to improve generalization by implementing equivariant filters using Fourier coefficients, which makes tranformation equivariant featuremaps .

The primary dataset used in this project is FashionMNIST, a dataset for image classification consisting of grayscale images of clothing items. The dataset contains 28×28 images across 10 different classes, such as shirts, trousers, and footwear. FashionMNIST is well suited for this study because the class label is equivaraint under rotations and relections, making it a useful for designing equivariant models. The dataset will initially be used with its standard training and test splits.

Two main models will be implemented and compared. The first is a baseline CNN, consisting of standard convolutional layers, nonlinearities, and pooling operations, serving as a baseline for performance and training behavior (potentially including data augmentation). The second is a steerable GCNN, implemented using escnn, which replaces standard convolutions with equivariant convolutions defined over different symmetry groups, such as the SO(2). Multiple symmetry configurations can be explored to study their impact on performance and robustness.

Both model types will be trained and optimized using a systematic hyperparameter sweep study, including learning rate, depth, width, and regularization. Final evaluation will be performed using the best-performing hyperparameter configurations.

**OBS: We have changed the dataset from pCAM to fashionMNIST, due to PCAM being difficult and large to work with.**

## Project structure

The directory structure of the project looks like this:

```txt
├── .devcontainer/                  # VS Code dev container config
├── .github/                        # GitHub Actions & workflows
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── .dvcignore                      # DVC ignores
├── .gcloudignore                   # Google Cloud ignore (Cloud Build)
├── configs/                        # Hydra configuration system
│   ├── train.yaml                  # Main training config
│   ├── config.yaml                 # Base config
│   ├── data/
│   │   └── fashion_mnist.yaml      # Dataset-specific configs
│   ├── model/
│   │   ├── cnn.yaml                # CNN model config
│   │   └── c8.yaml                 # Alternative model config
│   ├── trainer/
│   │   ├── cpu.yaml                # Trainer settings: CPU
│   │   ├── gpu.yaml                # Trainer settings: GPU
│   │   └── mps.yaml                # Trainer settings: Apple MPS
│   ├── callbacks/                  # Lightning callbacks configs
│   ├── logger/                     # Logging (CSV, wandb) configs
│   ├── paths/                      # Path configuration files
│   ├── hydra/                      # Hydra internal configs
│   ├── debug/                      # Debugging configs
│   ├── hparams_search/             # Hyperparameter sweep configs
│   ├── extras/                     # Extra configs
│   └── vertex_ai/                  # Vertex AI specific configs
│       └── config_cpu.yaml         # Vertex AI CPU job config
├── dockerfiles/                    # Docker builds
│   ├── api.dockerfile              # FastAPI server image
│   ├── train.dockerfile            # Training image
│   └── cloud.dockerfile            # Cloud training image
├── docs/                           # Documentation (mkdocs)
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── gcp/                            # Google Cloud configs
│   ├── cloudbuild.yaml
│   └── policy.yaml
├── models/                         # Trained/checkpoint model files
│   └── model.pth
├── notebooks/                      # Jupyter notebooks
├── profiler/                       # Profiling outputs
├── reports/                        # Reports & figures
│   └── figures/
├── src/                            # Source code
│   └── dtu_mlops_project/
│       ├── __init__.py
│       ├── api.py                  # FastAPI REST API
│       ├── apifile.py              # API utilities
│       ├── data.py                 # Data loading/preprocessing
│       ├── evaluate.py             # Evaluation code
│       ├── model.py                # Model architectures
│       ├── drift_detection_api.py  # Drift detection with Evidently
│       ├── setup_check.py          # Environment checks
│       ├── train.py                # Training script using Hydra
│       └── visualize.py            # Visualization tools
├── tests/                          # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── data_drift_detection.py
│   ├── drift_report.html
│   ├── input_output_data.py
│   ├── locustfile.py
│   ├── production_input_output.py
│   ├── README.md
│   ├── report.html
│   ├── testimage.jpg
│   ├── test_data.py
│   └── test_model.py
├── wandb/                          # WandB sweep metadata
│   └── sweep-fgjxye38/
├── .gitignore
<<<<<<< Updated upstream
├── .pre-commit-config.yaml   # Pre-commit hooks (linting, formatting)
├── AGENTS.md                 # Instructions for autonomous coding agents
├── data.dvc                  # DVC data versioning file
=======
├── .gitattributes
├── .pre-commit-config.yaml
├── .project-root
├── .python-version
├── AGENTS.md
>>>>>>> Stashed changes
├── LICENSE
├── README.md
├── data.dvc                        # DVC data versioning
├── image.png
├── pyproject.toml
├── report.html
├── tasks.py                        # Task definitions
└── uv.lock                         # Locked dependencies

```

## Component Descriptions (notable components)

### `src/dtu_mlops_project/`
- **train.py**: Main training script using PyTorch Lightning and Hydra for config management
- **data.py**: Data loading, preprocessing, and augmentation (RotatedFashionMNIST dataset)
- **model.py**: CNN and GCNN model architectures
- **evaluate.py**: Model evaluation and metrics computation
- **api.py**: FastAPI REST API for model inference
- **visualize.py**: Training visualization and analysis utilities
- **drift_detection_api.py**: Drift detection api using Evidently

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

### Tests
- **test_data.py**: Tests for dataset loading and preprocessing
- **test_model.py**: Model instantiation and forward-pass tests
- **test_api.py**: API endpoint tests
- **data_drift_detection.py**: Drift detection using Evidently
- **locustfile.py**: API load testing.


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
Fashion-MNIST downloads automatically to data on first run; no manual DVC pull is needed.

### Local Training
```sh
# Train with default config (CPU, Fashion MNIST)
uv run python src/dtu_mlops_project/train.py

# Train with custom config
uv run python src/dtu_mlops_project/train.py trainer=gpu model=c8

# Log with wandb
uv run python src/dtu_mlops_project/train.py logger=wandb

# Export trained model to ONNX format after training
uv run python src/dtu_mlops_project/train.py export_onnx=True

# Combined: Train and export to ONNX with custom hyperparameters
uv run python src/dtu_mlops_project/train.py trainer=gpu data.batch_size=64 export_onnx=True trainer.max_expochs=10
```

The exported ONNX model will be saved to: `logs/train/runs/<timestamp>/model.onnx`
ONNX models can be deployed to inference servers or used with ONNX Runtime.

### Hyperparameter Sweeps
```sh
# WandB sweep: 1 returns id, 2 runs agent.
1. uv run wandb sweep configs/hparams_search/wandb_sweep.yaml
2. uv run wandb agent --count 5 <sweep-id>

```

### Cloud Training (Vertex AI)
```sh
# Build and push Docker image to Artifact Registry

##TODO ÆNDRER TIL config folder istedet for root cloud
gcloud builds submit . --config configs/gcp/cloudbuild.yaml --substitutions=_IMAGE=api,_TAG=$(git rev-parse --short HEAD)

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


## Credits
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- [Machine Learning Operations](https://github.com/SkafteNicki/dtu_mlops)
