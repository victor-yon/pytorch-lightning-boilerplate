# Pytorch Lightning Boilerplate

A template to quickly set up a deep learning project with the library PyTorch and the framework Lightning.

The lightning framework claim to not required any boilerplate, but I believe that a boilerplate can still save some time
to set up the file structure and integrate some good practices such as configuration file.

## Quickstart template checklist

- [ ] Edit the [dataloader](datasets/project_dataset.py) to load your project data.
- [ ] Edit the [base model](models/base_model.py) to define your own model logic. If you plan to implement multiple
  models, extend this class.
- [ ] Check remaining `# TODO` comments in the code
- [ ] Remove this section from the [README](README.md)

---

<div align="center">    

# Project Name

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

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
# Create empty configuration file
touch config.yaml
```

Alternatively you can initialize a configuration file filled with default values using this
command: `python run.py --print_config > config.yaml`

### Run the main script

 ```bash
# If using a virtual environment, activate it
source venv/bin/activate

# Run the main script
python run.py
```

Note for IntelliJ users: The default run scripts should be automatically detected.

### Citation   

