from pathlib import Path

import pandas as pd

from configs.base_utils import DICTIONARY, SIGNREP, SPOTTINGS, VAL_DICTIONARY, ckpt_dir, lmdbs
from dataloaders.data_utils.classes import class_list
from gatle_ignite import base_config
from losses.loss_functions.dictionary_classifier import temperature

# 32 spottings per step; BATCH = 32, ACCUM = 1 is the same update (nothing normalises a batch).
BATCH = 16
ACCUM = 2


def get_config():
    classes = len(class_list(pd.read_csv(DICTIONARY), pd.read_csv(SPOTTINGS / "train.csv")))

    cfg = base_config()
    cfg.name = Path(__file__).stem
    cfg.project_name = "signmatch"
    cfg.logger_name = ["text"]
    cfg.save_dir = ckpt_dir(cfg.name)
    cfg.nproc_per_node = 1
    cfg.amp_dtype = "bf16"
    cfg.seed = 42

    cfg.main_runner = "trainer.stage1_trainer"
    cfg.model_name = "models.signmatch"
    cfg.model_params = {"chunk_size": 16, "num_classes": classes}
    cfg.signrep_checkpoint = str(SIGNREP)

    loader = {"lmdbs": lmdbs(), "bs": BATCH, "num_workers": 10, "pin_memory": False}
    cfg.aug_name = "augmentation.video"
    cfg.train_ds_name = "dataloaders.spottings"
    cfg.train_ds_params = {
        "spottings": str(SPOTTINGS / "train.csv"),
        "dictionary": str(DICTIONARY),
        "drop_last": True,
        "sampler_params": {
            "cls_name": "dataloaders.data_utils.head_sampler",
            "params": {"alpha": 0.5},
        },
        **loader,
    }
    cfg.valid_ds_name = "dataloaders.continuous_clips"
    cfg.valid_ds_params = {"spottings": str(SPOTTINGS / "val.csv"), **loader}
    cfg.test_ds_name = "dataloaders.continuous_clips"
    cfg.test_ds_params = {"spottings": str(SPOTTINGS / "test.csv"), **loader}
    ranking = {
        "ranking": {
            "cls_name": "metrics.class_ranking",
            "params": {"masked_to": str(VAL_DICTIONARY)},
        }
    }
    cfg.val_metrics = ranking
    cfg.tester_metrics = ranking
    cfg.every_val = 1
    cfg.score_name = "valid/DCG"

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "classifier": {
                "cls_name": "losses.loss_functions.stage1_classifier",
                "loss_params": {"temperature": temperature(classes, p=0.9)},
                "weight": 1.0,
            }
        }
    }
    cfg.optimizer_name = "optimizer.layer_decay"
    cfg.optimizer_params = {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "layer_decay": 0.85,
    }
    cfg.lr_scheduler = "scheduler.warmup_cosine"
    cfg.lr_scheduler_params = {
        "warmup_epochs": 1,
        "warmup_start_factor": 0.0,
        "end_factor": 0.03,
        "first_step": 1,
    }
    cfg.grad_clip_value = 1.0
    cfg.grad_clip_norm = 1.0
    cfg.accum_steps = ACCUM
    # train_length counts batches: 1,000 optimizer steps an epoch.
    cfg.max_epochs = 25
    cfg.train_length = 1000 * ACCUM
    return cfg
