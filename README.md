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
