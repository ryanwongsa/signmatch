from pathlib import Path

from configs.base_utils import SPOTTINGS, VAL_DICTIONARY, ckpt_dir, lmdbs
from dataloaders.data_utils.classes import num_classes
from gatle_ignite import base_config
from gatle_ignite.utils.checkpoints import find_best_checkpoint
from losses.loss_functions.dictionary_classifier import temperature


def get_config(stage1_run=ckpt_dir("stage1")):
    stage1_run = Path(stage1_run)
    best = find_best_checkpoint(stage1_run, prefix="valid_")
    if best is None:
        raise FileNotFoundError(f"no valid_best_result_checkpoint_*.pt in Stage 1 run {stage1_run}")
    manifest = stage1_run / "classes.csv"
    classes = num_classes(manifest)

    cfg = base_config()
    cfg.name = Path(__file__).stem
    cfg.project_name = "signmatch"
    cfg.logger_name = ["text"]
    cfg.save_dir = ckpt_dir(cfg.name)
    cfg.nproc_per_node = 1
    cfg.amp_dtype = "bf16"
    cfg.seed = 42

    cfg.main_runner = "trainer.stage2_trainer"
    cfg.model_name = "models.signmatch"
    cfg.model_params = {"chunk_size": 16, "num_classes": classes, "dictionary_head": "lstm"}
    cfg.stage1_checkpoint = str(best)

    loader = {"lmdbs": lmdbs(), "bs": 64, "num_workers": 10, "pin_memory": False}
    cfg.aug_name = "augmentation.video"
    cfg.train_ds_name = "dataloaders.dictionary"
    cfg.train_ds_params = {
        "manifest": str(manifest),
        "train": True,
        "shuffle": True,
        "drop_last": True,
        **loader,
    }
    cfg.dictionary_ds_name = "dataloaders.dictionary"
    cfg.dictionary_ds_params = {"manifest": str(VAL_DICTIONARY), "train": False, **loader}
    cfg.valid_ds_name = "dataloaders.continuous_clips"
    cfg.valid_ds_params = {"spottings": str(SPOTTINGS / "val.csv"), **loader}
    cfg.test_ds_name = "dataloaders.continuous_clips"
    cfg.test_ds_params = {"spottings": str(SPOTTINGS / "test.csv"), **loader}
    ranking = {"ranking": {"cls_name": "metrics.dictionary_ranking", "params": {}}}
    cfg.val_metrics = ranking
    cfg.tester_metrics = ranking
    cfg.every_val = 1
    cfg.score_name = "valid/DCG"

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "classifier": {
                "cls_name": "losses.loss_functions.dictionary_classifier",
                "loss_params": {"temperature": temperature(classes, p=0.9)},
                "weight": 1.0,
            }
        }
    }
    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 1e-4, "betas": (0.9, 0.999), "weight_decay": 1e-4}
    cfg.lr_scheduler = "scheduler.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1, "warmup_start_factor": 0.01, "end_factor": 0.1}
    cfg.grad_clip_value = 1.0
    cfg.grad_clip_norm = 1.0
    cfg.max_epochs = 10
    cfg.train_length = 1000
    return cfg
