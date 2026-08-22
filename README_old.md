# IT3385 MLOps Assignment

This project demonstrates an end-to-end MLOps workflow for developing, versioning, testing, deploying and monitoring machine learning applications.

Our team developed two machine learning solutions and integrated them into a common Streamlit web application:

- Predictive maintenance classification
- AI and data job salary prediction

The project uses **Poetry, Hydra, DVC, Git/GitHub, pytest, GitHub Actions, Docker, MLflow, Streamlit and Google Cloud Run** to support the machine learning development and deployment lifecycle.

---

## 1. Team Information

| Team Member | Dataset / Work Undertaken | Final Model |
|---|---|---|
| Darren Chor | Global AI & Data Jobs dataset - annual salary prediction (https://www.kaggle.com/datasets/mohankrishnathalla/global-ai-and-data-jobs-salary-dataset)| CatBoost Regressor |
| Javian Ng | Smart Manufacturing dataset - predictive maintenance (https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset) | XGBoost Classifier |

## 2. Project Structure

```text
mlops-assignment/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── darren/
│   ├── conf/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── javian/
│   ├── conf/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── webapp/
│   └── streamlit_app.py
│
├── .dockerignore
├── .dvcignore
├── .gcloudignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── poetry.lock
└── README.md
```

### Main Folders and Files

- `darren/` - Darren dataset, configuration, notebooks, trained model and tests.
- `javian/` - Javian dataset, configuration, notebooks, trained model and tests.
- `conf/` - Hydra configuration files for dataset paths and modelling settings.
- `data/raw/` - original datasets tracked using DVC.
- `data/processed/` - processed datasets tracked using DVC.
- `models/` - final trained machine learning pipelines used by the web application.
- `notebooks/` - EDA, model experimentation, tuning and evaluation notebooks.
- `tests/` - automated pytest tests used by the CI pipeline.
- `webapp/` - integrated Streamlit web application.
- `.github/workflows/` - GitHub Actions CI/CD workflow.
- `Dockerfile` - container configuration used for deployment.
- `pyproject.toml` - project dependencies managed using Poetry.
- `poetry.lock` - locked dependency versions for reproducibility.

---

## 3. Environment Setup

### 3.1 Requirements

The project uses:

- Python 3.11
- Conda
- Poetry
- Git
- DVC
- Google Cloud SDK for deployment

All team members use the same Python version to maintain a consistent development environment.

### 3.2 Create the Conda Environment

Create the environment:

```bash
conda create -n it3385-mlops python=3.11
```

Activate it:

```bash
conda activate it3385-mlops
```

Verify the Python version:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

## 4. Poetry Dependency Management

Poetry is used to manage project dependencies and provide a reproducible Python environment.

Check whether Poetry is installed:

```bash
poetry --version
```

If Poetry is not installed:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

Restart the terminal and install Poetry:

```bash
pipx install poetry
```

### Install Project Dependencies

Navigate to the repository root:

```bash
cd mlops-assignment
```

Install dependencies:

```bash
poetry install
```

The versions stored in `poetry.lock` are used to provide a consistent environment across team members and CI.

```

### Managing Dependencies

Add a normal dependency:

```bash
poetry add <package>
```

Add a development dependency:

```bash
poetry add --group dev <package>
```

Poetry automatically updates both `pyproject.toml` and `poetry.lock`.

### Verify the Environment

Run:

```bash
poetry run python -c "import pandas, numpy, sklearn, pycaret, mlflow, hydra, dvc; print('Environment OK')"
```

Expected:

```text
Environment OK
```

---

## 5. Hydra Configuration Management

Hydra is used to centralise configuration values and minimise hard coding during machine learning development.

Each of us maintains configuration files in our respective `conf/` directory.

Configuration values include items such as:

```yaml
data:
  raw_path: <raw-data-path>
  processed_path: <processed-data-path>

model:
  target: <target-variable>
  random_state: 42

training:
  test_size: 0.2
  fold: 10
```

The modelling code reads these settings through Hydra rather than repeatedly hard-coding dataset paths, targets and training parameters.

This makes configurations easier to maintain and modify without changing the main source code.

---

## 6. Data Version Control with DVC

DVC is used to version datasets without storing large CSV files directly in Git.

The project contains DVC metadata files such as:

```text
darren/data/raw/global_ai_jobs.csv.dvc
darren/data/processed/global_ai_jobs_preprocessed.csv.dvc

javian/data/raw/smart_manufacturing_data.csv.dvc
javian/data/processed/smart_manufacturing_processed.csv.dvc
```

A shared Google Drive folder is used as the DVC remote.

### Pull DVC-Tracked Data

After cloning the repository:

```bash
dvc pull
```

DVC reads the `.dvc` metadata and restores the appropriate dataset versions.

Example:

```text
global_ai_jobs.csv.dvc
        ↓
     dvc pull
        ↓
global_ai_jobs.csv
```

### Add or Update Data

The original files inside `data/raw/` should remain unchanged.

Processed data should be written to `data/processed/`.

To track a new or modified dataset:

```bash
dvc add <path-to-dataset>
```

Then upload it to the shared remote:

```bash
dvc push
```

Commit the updated DVC metadata:

```bash
git add .
git commit -m "Update DVC tracked data"
git push
```

### Data Version Control (DVC)

This project uses **DVC (Data Version Control)** to version and share datasets without storing the actual data files directly in GitHub.

A shared **Google Drive folder** is used as the DVC remote storage.

### DVC Remote Configuration

The shared remote is configured in `.dvc/config`:

```ini
[core]
    remote = myremote

['remote "myremote"']
    url = gdrive://1mD6bTYB9qNbT2rWKFndiomxlZTOkmIRi
```

Google Drive authentication requires a custom OAuth Client ID and Client Secret. These credentials are configured locally using:

```bash
dvc remote modify --local myremote gdrive_client_id <CLIENT_ID>
dvc remote modify --local myremote gdrive_client_secret <CLIENT_SECRET>
```

The Client ID and Client Secret are **not included in this repository for security reasons**.

ince the OAuth Client ID and Client Secret cannot be publicly included in this repository, users without access to the shared DVC Google Drive remote can place the required CSV files locally at the expected paths. (If access to the google drive is needed, request it from Darren the id, secrets and permission to the google drive)

---

## 7. Automated Testing

Automated tests are implemented using `pytest`.

Tests verify important project components such as:

- Hydra configuration files exist.
- Hydra YAML files contain valid configuration sections.
- Important model configuration values are correct.
- DVC metadata files exist.
- Required notebook directories exist.
- Required model directories exist.

Run the complete test suite locally using:

```bash
poetry run pytest
```

A successful run should complete all Darren and Javian tests without failures.

---

## 8. CI/CD Pipeline

GitHub Actions is used to implement **Continuous Integration** and **Continuous Delivery**.

### 8.1 Continuous Integration

Continuous Integration runs automatically for:

- pushes to `main`
- pull requests targeting `main`

The CI workflow performs:

```text
Checkout Repository
        ↓
Set Up Python 3.11
        ↓
Install Poetry
        ↓
Install Dependencies
        ↓
Run pytest
```

If any automated test fails, the workflow fails.

This prevents the Continuous Delivery stage from proceeding with code that has failed validation.

### 8.2 Continuous Delivery

After a successful push to `main`, the Continuous Delivery job runs only after the Continuous Integration job passes.

The workflow:

1. Checks out the tested source code.
2. Verifies the web application source.
3. Prepares the application files for release.
4. Creates a versioned GitHub Actions deployment artifact.

This ensures that a successfully tested version is prepared and ready for release.

The final production release remains manually controlled so that the team decides when a tested application version is deployed to Google Cloud Run. Automation of deployment was considered but wasn't implemented due to our app running on Google Cloud, so to prevent unnessacry charges everything we made edits to the streamlit app, we do it manually.

The workflow therefore follows:

```text
Feature Branch
      ↓
Pull Request / Push
      ↓
Continuous Integration
      ↓
Automated Tests
      ↓
Merge / Push to Main
      ↓
Continuous Delivery
      ↓
Release Artifact
      ↓
Manual Production Release
      ↓
Google Cloud Run
```

---

## 9. MLflow Experiment Tracking

MLflow is used during model development to track machine learning experiments.

MLflow records information including:

- experiment runs
- model parameters
- model performance metrics
- model artifacts
- selected model versions

The selected final models are registered in the MLflow Model Registry to provide a versioned record of the model used for deployment.

MLflow is used during development and is not required for each prediction made by the deployed Streamlit application.

---

## 10. Running the Web Application Locally

The project contains an integrated Streamlit application supporting both team members models.

From the project root, run:

```bash
streamlit run webapp/streamlit_app.py
```

Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

### Prediction Pages

| Page | Owner | Model | Prediction |
|---|---|---|---|
| Predictive Maintenance | Javian | XGBoost Classifier | Predicts whether machine maintenance is required and provides a risk probability |
| AI Job Salary Prediction | Darren | CatBoost Regressor | Predicts annual base salary in USD |

Both pages support:

- single predictions
- batch predictions using CSV uploads
- downloadable sample input files
- downloadable batch prediction results

### Required Model Files

The application expects the trained models at:

```text
javian/models/final_XGBClassifier.pkl
darren/models/final_salary_catboost_pipeline.pkl
```

The required model files must be available before starting the application.

---

## 13. User Guide

### 13.1 Accessing the Application

The deployed application can be accessed here https://mlops-assignment-987605116952.europe-west1.run.app/

---

### 13.2 AI Job Salary - Single Prediction

1. Open the **AI Job Salary Prediction** page.
2. Select the required categorical job information.
3. Enter the required numerical values.
4. Click **Predict Salary**.
5. The application validates the inputs.
6. The saved preprocessing and CatBoost pipeline automatically transforms the raw input.
7. The predicted annual salary is displayed in USD.

Example output:

```text
Estimated Annual Salary (USD)
$95,000
```

Invalid values are rejected and an appropriate validation message is displayed.

---

### 13.3 AI Job Salary - Batch Prediction

1. Open the **Batch Prediction** tab.
2. Download the provided sample CSV if required.
3. Prepare a CSV containing the required model features.
4. Upload the CSV.
5. The application validates the required columns and values.
6. Click **Run Batch Prediction**.
7. The model generates predictions for all valid rows.
8. Review the prediction results.
9. Download the resulting prediction CSV if required.

The salary model requires its 31 raw feature columns.

The complete saved pipeline automatically handles the preprocessing required by the CatBoost model.

---

### 13.4 Predictive Maintenance - Single Prediction

1. Open the **Predictive Maintenance** page.
2. Enter or select the required machine information.
3. Submit the input.
4. The application processes the data using the saved machine-learning pipeline.
5. The predicted maintenance outcome and associated probability/risk value are displayed.

---

### 13.5 Predictive Maintenance - Batch Prediction

1. Open the batch prediction section.
2. Download the sample CSV if required.
3. Prepare and upload a compatible CSV file.
4. Generate the batch predictions.
5. Review the results.
6. Download the generated predictions if required.

The maintenance batch data contains fields including:

```text
timestamp
machine_id
temperature
vibration
humidity
pressure
energy_consumption
```

Rows are grouped by machine and ordered by time so that the required historical features can be generated.

---

## 14. Deployment to Google Cloud Run

The integrated Streamlit application is containerised using Docker and deployed to **Google Cloud Run**.

Google Cloud Run provides a managed container environment with automatic scaling based on application traffic.

### 14.1 Initial Deployment

The Cloud Run service was initially created through the Google Cloud Console.

1. Open **Google Cloud Run**.
2. Click **Connect Repository**.
3. Select the GitHub repository using **Cloud Build**.
4. Configure the service name.
5. Select the deployment region. The project uses `europe-west1`.
6. Select **Request-based** billing.
7. Enable **Allow public access** so that the Streamlit application can be accessed through its public URL.
8. Create the Cloud Run service.

The repository contains a `Dockerfile`, which Cloud Build uses to build the application container before deploying it to Cloud Run.

The resulting container image is stored in Google Artifact Registry and deployed as a Cloud Run revision.

### 14.2 Redeploying an Updated Application

After the Cloud Run service has been created, updated versions of the application can be manually released from the project root.

Authenticate with Google Cloud if required:

```bash
gcloud auth login
```

Set the correct Google Cloud project:

```bash
gcloud config set project <project_id>
```

Redeploy the updated application:

```bash
gcloud run deploy <service_name> --source . --region
```

The `--source .` command sends the current project source to Google Cloud Build.

Because the repository contains a `Dockerfile`, Cloud Build uses it to build a new container image. The image is stored in Artifact Registry and deployed as a new Cloud Run revision.

A separate manual `docker build` and `docker push` step is not required.

### Deployment Workflow

```text
Updated and Tested Application
          ↓
Manual Redeployment
          ↓
Google Cloud Build
          ↓
Docker Image
          ↓
Artifact Registry
          ↓
New Cloud Run Revision
          ↓
Updated Streamlit Application
```

## 15. Monitoring

Google Cloud Run Observability is used to monitor the operational health of the deployed application.

The monitoring dashboard provides metrics including:

- request count
- request latency
- end-to-end request latency
- HTTP response status and errors
- container instance count
- billable container instance time
- application and container logs

This allows the team to monitor whether the deployed machine learning application remains available, responsive and operational after deployment.

Cloud Run also automatically scales the number of container instances according to traffic.

Model-specific monitoring such as feature drift, prediction drift or production RMSE could be added in a future production extension if ground-truth data becomes available.

---

## 16. URLs

### Deployed Web Application

https://mlops-assignment-987605116952.europe-west1.run.app

### Source Code Repository

https://github.com/1008tommy/mlops-assignment

