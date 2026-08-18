IT3385 MLOps Assignment initial environment setup

Before starting development, all team members should set up the same Python environment and install Poetry. This helps ensure that everyone is working with a consistent development environment.

## 1. Create the Conda Environment

All team members should use **Python 3.11**.

Create a new Conda environment:

**conda create -n it3385-mlops python=3.11**

Activate the environment:

**conda activate it3385-mlops**

Verify the Python version:

python --version

Expected output:

**Python 3.11.x**

Conda is used to provide a controlled and consistent Python version for all team members.

## 2. Install Poetry

Poetry will be used as the project's dependency manager.

First, check whether Poetry is already installed:

poetry --version

If Poetry is not installed, install pipx:

python -m pip install --user pipx

Add pipx to your PATH:

python -m pipx ensurepath

Close and reopen the terminal, then install Poetry:

pipx install poetry

Verify that Poetry is available:

poetry --version

You should see output similar to:

Poetry (version 2.x.x)
