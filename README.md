Diabetes-ml-case
==============================

This case has been prepared to analyze factors related to readmission as well as other outcomes pertaining to patients with diabetes.. There are not insights about the diabetes patients hospital readmissions. 

https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008#

**Requirements, assumptions and constraints**

Predict diabetes readmission patients:  

To the MVP we simplify the target to:  
1: if the patient was readmitted in less than 30   
0: for the rest  

**Business success criteria**  

Reduce cost of diabetes readmissions by 10% during the next 12 month after prediction get online after pacient trial phase.  

**Data mining success criteria**  
Auc-Roc above 50%  
Get the MVP in one day  

**Produce project plan**  
This project follow CRISP-DM, Scrum-DS and Gitflow methodologies.  

The milestones are organized following CRISP-DM Structure. The git-branch and commit are associaded to the task on those milestones.  

https://github.com/wiflore/Diabetes-ML-Case/projects/1




Project Organization
------------

    ├── LICENSE
    ├── Makefile                     <- Makefile with commands like `make data` or `make train` [Not use yet]
    ├── README.md                    <- The top-level README for developers using this project.
    ├── data                         <- A default Sphinx project; [OnPremise or Cloud]
    │   ├── feature-engineeres       <- Data with feature engineering
    │   ├── modeled                  <- Data Schema and table modeled
    │   ├── transformed              <- Data with transformations like standarization, outliers treatment, etc
    │   └── raw                      <- Data raw
    │
    ├── docs                         <- A default Sphinx project; [Not use yet]
    │
    ├── serialized-models            <- Trained and serialized models, model predictions, or model summaries
    │
    ├── research-notebooks           <- Jupyter notebooks with the initial research approach
    │
    ├── references                   <- Data dictionaries, manuals, etc. [Not use yet]
    │
    ├── reports                      <- Generated analysis as HTML, PDF, LaTeX, etc.[Not use yet]
    │   └── figures                  <- Generated graphics and figures to be used in reporting [Not use yet]
    │
    ├── requirements.txt             <- The requirements file for reproducing the analysis environment, e.g.
    │
    │
    ├── setup.py           <- makes project pip installable  so src can be imported  [Not use yet]
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data              <- Scripts to download or generate data [Not use yet]
    │   │   └── make_dataset.py
    │   │
    │   ├── features          <- Scripts to turn raw data into features for modeling [Not use yet]
    │   │   └── build_features.py
    │   │
    │   ├── models            <- Scripts to train models and then use trained models to make
    │   │   │                    predictions [Not use yet]
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   ├── utilities         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── helpers.py
    │   │   └── visuals.py
    │   │
    │   └── visualization     <- Scripts to create exploratory and results oriented visualizations [Not use yet]
    │       └── visualize.py
    │
    └── tox.ini               <- tox file with settings for running tox; see tox.readthedocs.io [Not use yet]
