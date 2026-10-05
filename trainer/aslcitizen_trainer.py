from gatle_ignite import BaseTrainer, to_device


class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        model_input = {"windows": batch["windows"], "lengths": batch["lengths"]}
        targets = {"video_id": batch["video_id"], "gloss": batch["gloss"], "split": batch["split"]}
        return to_device({"model_input": model_input, "targets": targets})
