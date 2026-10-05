import torch

from gatle_ignite import BaseTrainer, get_dataset, to_device
from models.signmatch import load_stage1

CLIP_SPLITS = ("valid", "test")


class Trainer(BaseTrainer):
    def build_model(self):
        model = super().build_model()
        load_stage1(model, self.cfg.stage1_checkpoint)
        return model

    def build_dataloaders(self):
        super().build_dataloaders()
        self.dls["dictionary"], self.infos["dictionary"] = get_dataset(
            self.cfg.dictionary_ds_name,
            self.cfg.dictionary_ds_params,
            self.transform,
            field="dictionary_ds_name",
        )
        return self.dls, self.infos

    def prep_batch(self, batch, split="train", **kwargs):
        if split in CLIP_SPLITS:
            model_input = {"clips": batch["clip"]}
            targets = {"label": batch["label"], "dictionary": self.dictionary}
        else:
            model_input = {"windows": batch["windows"], "lengths": batch["lengths"]}
            targets = {"label": batch["label"]}
        return to_device({"model_input": model_input, "targets": targets})

    def run_eval_engine(self, spec):
        if spec.engine_type in CLIP_SPLITS and self.engines.get(spec.key) is not None:
            self.dictionary = self.embed_dictionary()
        return super().run_eval_engine(spec)

    def embed_dictionary(self):
        self.model.eval()
        embeddings, ids = [], []
        for batch in self.dls["dictionary"]:
            x = self.prep_batch(batch, split="dictionary")
            autocast = torch.autocast(
                self.device_type, dtype=self.dtype, enabled=self.autocast_enabled
            )
            with torch.no_grad(), autocast:
                embeddings.append(self.forward(x["model_input"])["dictionary_z_features"].float())
            ids += batch["id"]
        print(f"embedded {len(ids)} dictionary videos")
        return {"embeddings": torch.cat(embeddings), "ids": ids}
