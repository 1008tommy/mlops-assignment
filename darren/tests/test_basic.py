from pathlib import Path


def test_config_exists():
    assert Path("darren/conf/config.yaml").exists()


def test_raw_data_dvc_exists():
    assert Path("darren/data/raw/global_ai_jobs.csv.dvc").exists()


def test_processed_data_dvc_exists():
    assert Path(
        "darren/data/processed/global_ai_jobs_preprocessed.csv.dvc"
    ).exists()


def test_notebooks_folder_exists():
    assert Path("darren/notebooks").exists()


def test_models_folder_exists():
    assert Path("darren/models").exists()
    
# Check that the Hydra configuration file is valid YAML
def test_config_is_valid_yaml():

    with open(
        "darren/conf/config.yaml",
        "r"
    ) as file:

        config = yaml.safe_load(file)

    assert config is not None
    assert "data" in config
    assert "model" in config
    assert "training" in config


# Check important model configuration values
def test_model_config():

    with open(
        "darren/conf/config.yaml",
        "r"
    ) as file:

        config = yaml.safe_load(file)

    assert config["model"]["target"] == "salary_usd"
    assert config["training"]["fold"] == 10