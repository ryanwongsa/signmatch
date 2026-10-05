import math

from ignite.engine import Events

from gatle_ignite.schedulers._utils import base_lr, steps_per_epoch


class WarmupCosine:
    def __init__(
        self,
        optimizer,
        lr,
        warmup_start_lr,
        end_lr,
        warmup_steps,
        total_steps,
        batches_per_epoch,
        accum_steps=1,
        first_step=0,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.batches_per_epoch = batches_per_epoch
        self.accum_steps = accum_steps
        self.first_step = first_step

    def lr_at(self, step):
        if step < self.warmup_steps:
            return (
                self.warmup_start_lr + (self.lr - self.warmup_start_lr) * step / self.warmup_steps
            )
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.end_lr + (self.lr - self.end_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def steps_taken(self, iteration):
        epoch, batch = divmod(iteration - 1, self.batches_per_epoch)
        steps_per_epoch = math.ceil(self.batches_per_epoch / self.accum_steps)
        return epoch * steps_per_epoch + batch // self.accum_steps

    def __call__(self, engine):
        lr = self.lr_at(self.steps_taken(engine.state.iteration) + self.first_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def get_scheduler(
    cfg,
    train_dl,
    optimizer,
    engines,
    warmup_epochs=1,
    warmup_start_factor=0.01,
    end_factor=0.1,
    first_step=0,
):
    batches = steps_per_epoch(cfg, train_dl)
    accum_steps = cfg.get("accum_steps", 1)
    per_epoch = math.ceil(batches / accum_steps)
    lr = base_lr(cfg, optimizer)
    scheduler = WarmupCosine(
        optimizer,
        lr=lr,
        warmup_start_lr=lr * warmup_start_factor,
        end_lr=lr * end_factor,
        warmup_steps=int(warmup_epochs * per_epoch),
        total_steps=cfg.max_epochs * per_epoch,
        batches_per_epoch=batches,
        accum_steps=accum_steps,
        first_step=first_step,
    )
    return scheduler, "trainer", Events.ITERATION_STARTED
