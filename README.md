# IT3385 MLOps Assignment - Initial Environment Setup

Before starting development, all team members should set up the same Python environment and install Poetry. This helps ensure that everyone is working with a **consistent development environment**.

---

## 1. Create the Conda Environment

All team members should use **Python 3.11**.

Create a new Conda environment:

```bash
conda create -n it3385-mlops python=3.11
```

Activate the environment:

```bash
conda activate it3385-mlops
```

Verify the Python version:

```bash
python --version
```

Expected output:

```
Python 3.11.x
```

> Conda is used to provide a controlled and consistent Python version for all team members.

---

## 2. Install Poetry

Poetry will be used as the project's dependency manager.

First, check whether Poetry is already installed:

```bash
poetry --version
```

If Poetry is **not** installed, install `pipx`:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

Close and reopen the terminal, then install Poetry:

```bash
pipx install poetry
```

Verify that Poetry is available:

```bash
poetry --version
```

You should see output similar to:

```
Poetry (version 2.x.x)
```

---

## 3. Install the Project Dependencies

After cloning or pulling the repository, activate the Conda environment:

```bash
conda activate it3385-mlops
```

Navigate to the project root where `pyproject.toml` and `poetry.lock` are located:

```bash
cd mlops-assignment
```

Install all dependencies:

```bash
poetry install
```

If a `poetry.lock` file already exists, Poetry will install the dependency versions recorded in the lock file.

Expected output will be similar to:

```
Installing dependencies from lock file

Package operations: XX installs, X updates, X removals

  - Installing pandas (...)
  - Installing numpy (...)
  - Installing scikit-learn (...)
  - Installing pycaret (...)
  - Installing mlflow (...)
  ...
```

If all dependencies are already installed, Poetry may instead display:

```
Installing dependencies from lock file

No dependencies to install or update
```

> **Important**
>
> This project is an **application repository** rather than a Python package. Add the following to `pyproject.toml`:
>
> ```toml
> [tool.poetry]
> package-mode = false
> ```
>
> This tells Poetry to manage the dependencies without trying to install `mlops-assignment` itself as a Python package.

---

## 4. Adding New Dependencies

Poetry dependencies are managed using the `pyproject.toml` file.

> Do **not** normally edit dependency versions manually. Use `poetry add` so that Poetry can automatically resolve compatible versions and update both:
> - `pyproject.toml`
> - `poetry.lock`

### Add a New Library

For example, to add Streamlit:

```bash
poetry add streamlit
```

To add a library with a version constraint:

```bash
poetry add "matplotlib>=3.7,<3.8"
```

For example, the project currently uses PyCaret 3.3.2:

```bash
poetry add "pycaret==3.3.2"
```

After running `poetry add`, Poetry will:

1. Check the new dependency.
2. Check whether it is compatible with existing dependencies.
3. Resolve the required package versions.
4. Install the package.
5. Update `pyproject.toml`.
6. Update `poetry.lock`.

Expected output will look similar to:

```
Updating dependencies
Resolving dependencies... (X.Xs)

Package operations: X installs, X updates, X removals

  - Installing package-name (...)

Writing lock file
```

---

## 5. Adding Development Dependencies

Libraries that are only needed during development or testing should be placed in a separate **development dependency group**.

For example:

```bash
poetry add --group dev pytest
poetry add --group dev ruff
```

These tools can be used for automated testing and code quality checks later in the CI/CD pipeline.

---

## 6. Removing a Dependency

If a package is no longer required, remove it using:

```bash
poetry remove package-name
```

For example:

```bash
poetry remove streamlit
```

Poetry will automatically update both `pyproject.toml` and `poetry.lock`.

---

## 7. Check Installed Dependencies

To view the packages installed and managed by Poetry:

```bash
poetry show
```

To check a specific dependency:

```bash
poetry show pycaret
```

To verify the Poetry configuration:

```bash
poetry check
```

---

## 8. Verify the Complete Initial Setup

Every team member should be able to run:

```bash
conda activate it3385-mlops
python --version
poetry --version
poetry install
```

Then test the main dependencies:

```bash
poetry run python -c "import pandas, numpy, sklearn, pycaret, mlflow, hydra, dvc; print('Environment OK')"
```

Expected output:

```
Environment OK
```
## 9. Updating DVC-Tracked Data

The original dataset in `data/raw/` should **remain unchanged**. Any cleaning or transformation should produce a new dataset inside `data/processed/`.

If the processed dataset changes, re-add it to DVC:

```bash
dvc add darren/data/processed/global_ai_jobs_cleaned.csv
```

> Never edit or overwrite files in `data/raw/` directly. Always write cleaned or transformed output to `data/processed/`, then track the update with `dvc add`.

## Data Version Control (DVC)

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

If the required OAuth credentials are unavailable, the project can still be run using the dataset stored locally at the expected file path instead of retrieving it from the DVC remote.

### Pulling Data from DVC

To retrieve the DVC-tracked datasets from the shared Google Drive remote:

```bash
dvc pull
```

DVC uses the corresponding `.dvc` metadata files to identify and restore the correct dataset version.

For example:

```text
darren/data/raw/global_ai_jobs.csv.dvc
        ↓
     dvc pull
        ↓
darren/data/raw/global_ai_jobs.csv
```

The project then accesses the restored dataset using the paths defined in the Hydra configuration:

```yaml
data:
  raw_path: darren/data/raw/global_ai_jobs.csv
  processed_path: darren/data/processed/global_ai_jobs_cleaned.csv
```

### Pushing Data to DVC

To upload locally tracked data to the shared Google Drive remote:

```bash
dvc push
```

### Adding or Updating DVC-Tracked Data

When a new dataset or updated dataset needs to be tracked by DVC, use:

```bash
dvc add <path-to-dataset>
```

For example, to track the processed dataset:

```bash
dvc add darren/data/processed/global_ai_jobs_cleaned.csv
```

Then upload the latest version to the shared DVC remote:

```bash
dvc push
```

The generated or updated `.dvc` metadata file should then be committed to Git:

```bash
git add .
git commit -m "Update DVC tracked data"
git push
```

### Running Without DVC Access

Since the OAuth Client ID and Client Secret cannot be publicly included in this repository, users without access to the shared DVC Google Drive remote can place the required CSV files locally at the expected paths. (If access to the google drive is needed, request it from Darren the id, secrets and permission to the google drive)

For Darren dataset:

```text
darren/data/raw/global_ai_jobs.csv
darren/data/processed/global_ai_jobs_cleaned.csv
```

The Hydra configuration will then load the local files normally:

```yaml
data:
  raw_path: darren/data/raw/global_ai_jobs.csv
  processed_path: darren/data/processed/global_ai_jobs_cleaned.csv
```

DVC is used for dataset versioning and team data sharing, while the project can still be executed using local CSV files when access to the shared DVC remote is unavailable.

---

## Running the Web App

The repository includes a **Streamlit** web app with two prediction pages — Javian's predictive maintenance classifier and Darren's salary regressor. The app loads the trained models and datasets directly from disk, so run it after `dvc pull` (or after placing the files locally, see [DVC](#data-version-control-dvc)).

Start the app from the project root:

```bash
streamlit run webapp/streamlit_app.py
```

| Page | Owner | Model | Prediction |
|------|-------|-------|------------|
| Maintenance | Javian | XGBoost classifier | Whether a machine needs maintenance, with a risk probability |
| Salary | Darren | CatBoost regressor | Annual base salary in USD |

Both pages support **single** prediction (via input widgets) and **batch** prediction (upload a CSV and download the results). Each page also provides a downloadable sample input file.

### Required local files

The app reads these paths at runtime:

```text
javian/models/final_XGBClassifier.pkl
javian/data/raw/smart_manufacturing_data.csv
darren/models/final_salary_catboost_pipeline.pkl
darren/data/raw/global_ai_jobs.csv
```

> The models are loaded with `joblib`. The raw CSVs are used for the sample input files (and, for the maintenance page, the feature-clipping bounds), so they must be present even if you only run single prediction.

### Batch input format

- **Maintenance page** — CSV with columns `timestamp`, `machine_id`, `temperature`, `vibration`, `humidity`, `pressure`, `energy_consumption`. Rows are grouped per machine and ordered by time; each machine needs at least two rows so the lag features can be built.
- **Salary page** — CSV with the model's 31 feature columns (the raw `global_ai_jobs.csv` or a subset works). Categorical values must use the same labels as the training data.

## 11. Deployment (Google Cloud Run)

The web application is containerised with Docker and deployed to **Google Cloud Run**, a fully managed serverless platform that runs containers and auto-scales based on traffic (including scaling to zero when idle).

### Prerequisites
- Google Cloud SDK (`gcloud`) installed and authenticated
- A GCP project with billing enabled
- Cloud Run API and Artifact Registry API enabled

### Steps to deploy

1. **Authenticate and set the project**
```bash
   gcloud auth login
   gcloud config set project <PROJECT_ID>
```

2. **Build and push the container image** (using Cloud Build, no local Docker needed)
```bash
   gcloud builds submit --tag asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/maintenance-app
```

3. **Deploy to Cloud Run**
```bash
   gcloud run deploy maintenance-app \
     --image europe-west1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/maintenance-app \
     --platform managed \
     --region asia-southeast1 \
     --allow-unauthenticated \
     --port 8080
```

4. **Access the deployed app**
   Cloud Run returns a public HTTPS URL on successful deployment, e.g.: https://mlops-assignment-987605116952.europe-west1.run.app
