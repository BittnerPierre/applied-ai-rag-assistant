import configparser
from pathlib import Path


def load_config():
    base_dir = Path(__file__).resolve().parent.parent  # Chemin du dossier projet
    config_path = base_dir.parent / 'conf' / 'config.ini'

    config = configparser.ConfigParser()
    config.read(config_path)
    return config
