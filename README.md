<div align="center">

# Pytorch Lightning Boilerplate

</div>

A template to quickly set up a deep learning project with the library PyTorch and the framework Lightning.

Lightning authors claim that their framework replaces a boilerplate,
but I believe that a boilerplate still allows saving time and helps to maintain good practices.

This template integrates the following features:

* Interchangeable models and datasets (using standard [Lightning modules](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html))
* Configuration file and CLI to change and keep track of most of the meta-parameters (
  using [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html))
* Metrics integration (using [Lightning metrics](https://pytorch-lightning.readthedocs.io/en/stable/extensions/metrics.html))
* Preconfigured local logging (console and file) with [Loguru](https://github.com/Delgan/loguru), and optional remote
  logging ([Weights & Biases](https://wandb.ai/) or [MLFlow](https://mlflow.org))
* Basic unit tests to quickly detect implementation errors

## Quickstart template checklist

- [ ] Follow the "__How to run__" section of the [README](README.md).
- [ ] Edit the [dataloader](src/datasets/project_dataset.py) to load your project data.
- [ ] Edit the [base model](src/models/base_model.py) to define your own model logic. If you plan to implement multiple
  models, extend this class.
- [ ] Check remaining `# TODO` comments in the code.
- [ ] Run the unit tests ([./test](./test)) to validate the current implementation.
- [ ] Remove this section from the [README](README.md), and complete the project name and description below.

---

[//]: # (TODO: Remove section above)
<div align="center">

# Project Name

[//]: # (TODO: Set up badges with https://shields.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-informational)](https://docs.python.org/3/whatsnew/3.11.html)
[![Article](https://img.shields.io/badge/Article-doi.xxx-success)](https://www.doi.org/)
[![Conference](https://img.shields.io/badge/Conference-doi.xxx-success)](https://www.doi.org/)

</div>

## Description

[//]: # (TODO: Add a description of the project)
Project description.

## How to run

### Install dependencies

```bash
# Clone project
git clone https://github.com/<github_name>/<project_name>.git
cd <project_name>

# Optional but recommended: create a virtual environment with python 3.11
python --version # Should be 3.11 or higher
python -m venv venv
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
 ```   

### Create configuration file

```bash
# Create an empty configuration file, or ...
touch config.yaml
# ... initialise with default values and documentation
python src/run.py --print_config=comments > config.yaml
```

### Run the main script

 ```bash
# If using a virtual environment, activate it
source venv/bin/activate

# Run the main script
python run.py
```

> **Note for IntelliJ users:**
> The IDE should automatically detect the default run configuration.

## Files structure

```graphql
  ├─ out/ - # Output directory (created automatically)
  │
  ├─ src/ - # Contain the source code
  │  ├─ datasets/ - # Dataset and dataloader impelmentations
  │  ├─ models/ - # Model impelmentations (including baselines)
  │  ├─ plots/ - # Every code related to plot generation
  │  ├─ utils/ - # Utility functions used in various places
  │  └─ run.py - # Main script to run the project
  │
  ├─ tests/ - # Basic unit tests to quickly validate the implementation
  │
  ├─ README.md - # General documentation (this page)
  ├─ requirements.txt - # List of pip packages required for this project
  └─ config.yaml - # Configuration file (has to be created manually)
```

## Citation

[//]: # (TODO: Add citation information and/or create a CITATION.cff file, see https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)

```bibtex
@software{<main_author>_<year>,
  author = {<last_name_1>, <first_name_1> and <last_name_2>, <first_name_2>},
  doi = {<doi_url>},
  title = {{<Project title>}},
  url = {https://github.com/<github_name>/<project_name>},
  version = {1.0.0},
  year = {<year>},
  month = {<month>}
}
```