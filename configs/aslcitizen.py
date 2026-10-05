from pathlib import Path

from configs.base_utils import ASL_CITIZEN, ckpt_dir
from gatle_ignite import base_config


def get_config():
    cfg = base_config()
    cfg.name = Path(__file__).stem
    cfg.project_name = "signmatch"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "trainer.aslcitizen_trainer"
    cfg.model_name = "models.signmatch"
    cfg.model_params = {"chunk_size": 16}
    cfg.amp_dtype = "fp32"

    aslcitizen = {
        "splits_dir": str(ASL_CITIZEN / "splits"),
        "clips_dir": str(ASL_CITIZEN / "clips"),
        "bs": 16,
        "num_workers": 8,
    }
    cfg.test_ds_name = "dataloaders.aslcitizen"
    cfg.test_ds_params = aslcitizen
    cfg.every_test = 1
    cfg.tester_metrics = {
        "retrieval": {"cls_name": "metrics.retrieval", "params": {"out_dir": None}}
    }

    # Required by the framework. Scoring never iterates, optimises or computes a loss.
    cfg.train_ds_name = "dataloaders.aslcitizen"
    cfg.train_ds_params = dict(aslcitizen)
    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "classifier": {
                "cls_name": "losses.loss_functions.dictionary_classifier",
                "loss_params": {"temperature": 1.0},
                "weight": 1.0,
            }
        }
    }
    return cfg
