# Pytorch Lightning Boilerplate

A template to quickly set up a deep learning project with the library PyTorch and the framework Lightning.

Lightning authors claim that their framework replaces a boilerplate,
but I believe that a boilerplate still allows saving time and helps to maintain good practices.

This template integrates the following features:

* Interchangeable models and datasets (using standard [Lightning modules](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html))
* Configuration file to change and keep track of most of the meta-parameters (using [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html))
* Metrics integration (using [Lightning metrics](https://pytorch-lightning.readthedocs.io/en/stable/extensions/metrics.html))

## Quickstart template checklist

- [ ] Edit the [dataloader](src/datasets/project_dataset.py) to load your project data.
- [ ] Edit the [base model](src/models/base_model.py) to define your own model logic. If you plan to implement multiple
  models, extend this class.
- [ ] Check remaining `# TODO` comments in the code.
- [ ] Remove this section from the [README](README.md), and complete project name and description.

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

# Optional but recommended: create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
 ```   

### Create configuration file

```bash
# Create an empty configuration file
touch config.yaml
```

> **Note:**
> Alternatively you can initialize a configuration file filled with default values with this
> command: `python run.py --print_config=comments > config.yaml`

### Run the main script

 ```bash
# If using a virtual environment, activate it
source venv/bin/activate

# Run the main script
python run.py
```

> **Note for IntelliJ users:**
> The default run scripts should be automatically detected by the IDE.

## Files structure

```graphql
  ├─ out/ - # Output directory (created automatically)
  ├─ src/ - # Contain the source code
  │  ├─ datasets/ - # Dataset and dataloader impelmentations
  │  ├─ models/ - # Model impelmentations (including baselines)
  │  ├─ plots/ - # Every code related to plot generation
  │  ├─ utils/ - # Utility functions used in various places
  │  │  ├─ cli_config.py - # Command line interface and configuration file handling
  │  │  ├─ setup_env.py - # Place to initialize the environement for various components
  │  │  └─ trainer.py - #  Logic related to the model training
  │  └─ run.py - # Main script to run the project
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