# PG Room Rent Prediction – MLOps

## Project Information

| Field | Details |
|---|---|
| Course | SCSE3040 – MLOps |
| Group ID | G156 |
| Problem ID | MP265 |
| Project | PG Room Rent Prediction |
| Location | Surat, Gujarat |

## Project Overview

This project focuses on developing a Machine Learning Operations (MLOps) pipeline for predicting PG room rents in Surat.

The project involves data quality analysis, data preprocessing, machine learning model development, experiment tracking, deployment, and model monitoring.

## Team Members

| Name | Enrollment Number |
|---|---|
| Shubhangi Singh | S24CSEU2686 |
| Mukul Dabas | S24CSEU2497 |
| Neil David | S24CSEU1056 |
| Ashutosh Gupta Kumar | S24CSEU2477 |

## Technology Stack

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Version Control:** Git and GitHub
- **Data Version Control:** DVC
- **Experiment Tracking:** MLflow
- **Deployment:** FastAPI, Docker
- **Orchestration:** Kubernetes

## Dataset Description

The project includes the following datasets:

| Dataset | Description |
|---|---|
| train.csv | Training dataset |
| later.csv | Future data for monitoring |
| holdout_features.csv | Holdout features for prediction |

## Project Stages

### Stage 1: Repository and Data Quality
- GitHub repository setup
- DVC initialization
- Dataset exploration
- Identification of data-quality issues

### Stage 2: Data Processing and Model Development
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- MLflow experiment tracking

### Stage 3: Model Deployment
- FastAPI integration
- Docker containerization
- GitHub Actions CI/CD
- Kubernetes deployment

### Stage 4: Monitoring and Maintenance
- Data and model drift monitoring
- Model retraining
- Prediction monitoring
- Final pipeline validation

## Repository Structure

```text
Mlops-/
├── data/
│   └── raw/
├── src/
│   └── mp265/
├── tests/
├── findings.yaml
├── README.md
└── requirements.txt
```

## Objective

To build a reliable, reproducible, and maintainable end-to-end MLOps pipeline for PG room rent prediction.
