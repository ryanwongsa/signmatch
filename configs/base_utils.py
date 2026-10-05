import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
ASL_CITIZEN = DATA_ROOT / "ASL_Citizen"
SPOTTINGS = DATA_ROOT / "spottings"
DICTIONARY = DATA_ROOT / "dictionary.csv"
VAL_DICTIONARY = DATA_ROOT / "val_dictionary.csv"
SIGNREP = DATA_ROOT / "signrep.pt"


def ckpt_dir(name):
    return str(CHECKPOINTS / name)


def lmdbs():
    return {path.name: str(path) for path in sorted((DATA_ROOT / "lmdbs").iterdir())}
