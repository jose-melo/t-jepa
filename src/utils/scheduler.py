import math


class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.0,
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def state_dict(self):
        return {
            "start_lr": self.start_lr,
            "ref_lr": self.ref_lr,
            "final_lr": self.final_lr,
            "warmup_steps": self.warmup_steps,
            "T_max": self.T_max,
            "_step": self._step,
        }

    def load_state_dict(self, state_dict):
        self.start_lr = state_dict["start_lr"]
        self.ref_lr = state_dict["ref_lr"]
        self.final_lr = state_dict["final_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.T_max = state_dict["T_max"]
        self._step = state_dict["_step"]

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd

    def get_last_wd(self):
        return [group["weight_decay"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "ref_wd": self.ref_wd,
            "final_wd": self.final_wd,
            "T_max": self.T_max,
            "_step": self._step,
        }

    def load_state_dict(self, state_dict):
        self.ref_wd = state_dict["ref_wd"]
        self.final_wd = state_dict["final_wd"]
        self.T_max = state_dict["T_max"]
        self._step = state_dict["_step"]
