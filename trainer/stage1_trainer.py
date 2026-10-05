from pathlib import Path

import torch

from gatle_ignite import BaseTrainer, to_device
from models.signmatch import load_signrep

CLASSES = "classes.csv"


class Trainer(BaseTrainer):
    def build_model(self):
        model = super().build_model()
        load_signrep(model, self.cfg.signrep_checkpoint)
        model.projection_dict.requires_grad_(False)
        return model

    def build_dataloaders(self):
        super().build_dataloaders()
        classes = self.infos["train"]["classes"]
        write_classes(classes, Path(self.cfg.save_dir) / CLASSES)
        for split in ("valid", "test"):
            if self.infos.get(split) is not None:
                self.infos[split]["classes"] = classes
        return self.dls, self.infos

    def prep_batch(self, batch, split="train", **kwargs):
        model_input = {"clips": batch["clip"]}
        x = to_device({"model_input": model_input, "targets": {"label": batch["label"]}})
        x["model_input"]["classify"] = True
        return x

    def _clip_grads(self, model):
        if self.grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip_value)
        if self.grad_clip_norm is not None:
            params = [p for name, p in model.named_parameters() if "confusion" not in name]
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)



def write_classes(classes, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    classes.to_csv(path, index=False)
    print(f"wrote {len(classes)} classes to {path}")

