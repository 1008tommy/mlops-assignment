from pathlib import Path
import yaml

def test_config_exists():
    assert Path("javian/conf/config.yaml").exists()


def test_raw_data_dvc_exists():
    assert Path("javian/data/raw/smart_manufacturing_data.csv.dvc").exists()


def test_processed_data_dvc_exists():
    assert Path(
        "javian/data/processed/smart_manufacturing_processed.csv.dvc"
    ).exists()


def test_notebooks_folder_exists():
    assert Path("javian/notebooks").exists()


def test_models_folder_exists():
    assert Path("javian/models").exists()
    
# Check that the Hydra configuration file is valid YAML
def test_config_is_valid_yaml():

    with open(
        "javian/conf/config.yaml",
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
        "javian/conf/config.yaml",
        "r"
    ) as file:

        config = yaml.safe_load(file)

    assert config["model"]["target"] == "maintenance_required"
    assert config["training"]["fold"] == 10