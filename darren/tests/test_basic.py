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