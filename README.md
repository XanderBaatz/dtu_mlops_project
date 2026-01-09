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
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
