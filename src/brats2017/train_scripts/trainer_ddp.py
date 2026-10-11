import math
import os
import warnings
from copy import deepcopy
from typing import Dict, Mapping, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.segmentation_metrics import SlidingWindowInference

try:
    # Model file currently placed at:
    # <SREFORMER_BRATS_ROOT>\segformer3d_constructive_primitive_field_extreme.py
    from segformer3d_constructive_primitive_field_extreme import (
        constructive_primitive_field_aux_loss,
    )
except ImportError:
    # Optional fallback when the model is later moved under architectures/.
    from architectures.segformer3d_constructive_primitive_field_extreme import (
        constructive_primitive_field_aux_loss,
    )


warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated.*",
)


class Segmentation_Trainer:
    """
    Trainer for the CAPF-Extreme model.

    Total objective:
        L_total = L_main + lambda_capf(epoch) * L_CAPF

    L_main:
        The project's existing segmentation criterion, normally Dice loss.

    L_CAPF:
        Primitive-field supervision
        + image-evidence supervision
        + boundary supervision
        + uncertainty-to-boundary alignment
        + primitive sparsity / overlap / carving / hierarchy regularization.

    The CAPF coefficient is progressively activated to prevent the analytic
    primitive field from dominating the image-evidence branch at initialization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        accelerator,
        lr_scheduler=None,
        warmup_scheduler=None,
        training_scheduler=None,
        config: Dict = None,
    ) -> None:
        if config is None:
            raise ValueError("Segmentation_Trainer requires a non-empty config dictionary.")

        self.accelerator = accelerator
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.device = self.accelerator.device
        self.non_blocking = True

        self.lr_scheduler = lr_scheduler
        self.warmup_scheduler = warmup_scheduler
        self.train_scheduler = training_scheduler

        training_cfg = config["training_parameters"]
        self.grad_accumulate_steps = int(training_cfg["grad_accumulate_steps"])
        self.print_every = int(training_cfg["print_every"])
        self.calculate_metrics = bool(training_cfg["calculate_metrics"])
        self.num_epochs = int(training_cfg["num_epochs"])
        self.cutoff_epoch = int(training_cfg["cutoff_epoch"])
        self.checkpoint_save_dir = training_cfg["checkpoint_save_dir"]

        clip_cfg = config.get("clip_gradients", {})
        self.clip_gradients_enabled = bool(clip_cfg.get("enabled", False))
        self.clip_gradients_value = float(
            clip_cfg.get("clip_gradients_value", 1.0)
        )

        self.capf_cfg = dict(config.get("capf_loss", {}))
        self.capf_enabled = bool(self.capf_cfg.get("enabled", True))
        self.lambda_capf = float(self.capf_cfg.get("lambda_capf", 1.0))
        self.capf_start_epoch = int(self.capf_cfg.get("start_epoch", 0))
        self.capf_ramp_epochs = max(
            0, int(self.capf_cfg.get("ramp_epochs", 0))
        )
        self.capf_ramp_type = str(
            self.capf_cfg.get("ramp_type", "cosine")
        ).lower()

        ema_cfg = config["ema"]
        self.use_ema = bool(ema_cfg["enabled"])
        self.ema_decay = float(ema_cfg["ema_decay"])
        self.val_ema_every = int(ema_cfg["val_ema_every"])

        if self.use_ema:
            self.ema_model = deepcopy(self.model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
        else:
            self.ema_model = None

        self.best_dice = 0.0
        self.current_epoch = 0

        roi = tuple(config["sliding_window_inference"]["roi"])
        sw_batch_size = int(
            config["sliding_window_inference"]["sw_batch_size"]
        )
        self.sliding_window_inferer = SlidingWindowInference(
            roi=roi,
            sw_batch_size=sw_batch_size,
        )

    # ==============================================================================================
    # Device / EMA
    # ==============================================================================================

    def _move_to_device(self, x):
        if torch.is_tensor(x):
            return x.to(
                self.device,
                non_blocking=self.non_blocking,
            )

        if isinstance(x, dict):
            return {
                key: self._move_to_device(value)
                for key, value in x.items()
            }

        if isinstance(x, list):
            return [self._move_to_device(value) for value in x]

        if isinstance(x, tuple):
            return tuple(self._move_to_device(value) for value in x)

        return x

    def _move_batch_to_device(self, batch):
        return self._move_to_device(batch)

    @torch.no_grad()
    def _update_ema_model(self):
        if not self.use_ema:
            return

        model_state = self.accelerator.unwrap_model(self.model).state_dict()
        ema_state = self.ema_model.state_dict()

        for key in ema_state.keys():
            source = model_state[key].detach()
            target = ema_state[key]

            if torch.is_floating_point(target):
                target.mul_(self.ema_decay).add_(
                    source,
                    alpha=1.0 - self.ema_decay,
                )
            else:
                target.copy_(source)

    # ==============================================================================================
    # CAPF loss
    # ==============================================================================================

    def _capf_ramp_factor(self, epoch: int) -> float:
        """
        Returns a factor in [0, 1].

        Epochs before start_epoch:
            0

        During ramp:
            linear or cosine increase

        After ramp:
            1
        """
        if not self.capf_enabled:
            return 0.0

        if epoch < self.capf_start_epoch:
            return 0.0

        if self.capf_ramp_epochs <= 0:
            return 1.0

        progress = (
            epoch - self.capf_start_epoch + 1
        ) / float(self.capf_ramp_epochs)
        progress = max(0.0, min(1.0, progress))

        if self.capf_ramp_type == "linear":
            return progress

        if self.capf_ramp_type == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * progress)

        raise ValueError(
            f"Unsupported capf_loss.ramp_type={self.capf_ramp_type!r}. "
            "Use 'linear' or 'cosine'."
        )

    def _compute_total_loss(
        self,
        predicted: torch.Tensor,
        labels: torch.Tensor,
        aux_outputs: Mapping[str, object],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_seg = self.criterion(predicted, labels)

        zero = loss_seg.new_zeros(())
        capf_aux = zero
        capf_details: Dict[str, torch.Tensor] = {}

        capf_scale = (
            self.lambda_capf * self._capf_ramp_factor(epoch)
        )

        if self.capf_enabled:
            capf_aux, capf_details = (
                constructive_primitive_field_aux_loss(
                    aux=aux_outputs,
                    target=labels,
                    primitive_supervision_weight=float(
                        self.capf_cfg.get(
                            "primitive_supervision_weight",
                            0.20,
                        )
                    ),
                    evidence_supervision_weight=float(
                        self.capf_cfg.get(
                            "evidence_supervision_weight",
                            0.10,
                        )
                    ),
                    boundary_weight=float(
                        self.capf_cfg.get("boundary_weight", 0.20)
                    ),
                    gate_sparsity_weight=float(
                        self.capf_cfg.get(
                            "gate_sparsity_weight",
                            1e-3,
                        )
                    ),
                    positive_overlap_weight=float(
                        self.capf_cfg.get(
                            "positive_overlap_weight",
                            2e-3,
                        )
                    ),
                    negative_overlap_weight=float(
                        self.capf_cfg.get(
                            "negative_overlap_weight",
                            2e-3,
                        )
                    ),
                    negative_outside_weight=float(
                        self.capf_cfg.get(
                            "negative_outside_weight",
                            5e-3,
                        )
                    ),
                    hierarchy_weight=float(
                        self.capf_cfg.get(
                            "hierarchy_weight",
                            1e-2,
                        )
                    ),
                    uncertainty_weight=float(
                        self.capf_cfg.get(
                            "uncertainty_weight",
                            1e-3,
                        )
                    ),
                )
            )

        total_loss = loss_seg + capf_scale * capf_aux

        if not torch.isfinite(total_loss):
            numeric = {
                "loss_seg": float(loss_seg.detach().cpu()),
                "capf_aux": float(capf_aux.detach().cpu()),
                "capf_scale": capf_scale,
            }
            raise FloatingPointError(
                f"Non-finite CAPF training loss detected: {numeric}"
            )

        details: Dict[str, torch.Tensor] = {
            "total": total_loss.detach(),
            "seg": loss_seg.detach(),
            "capf_unscaled": capf_aux.detach(),
            "capf_scale": loss_seg.new_tensor(capf_scale),
        }

        for key, value in capf_details.items():
            if torch.is_tensor(value):
                details[key] = value.detach()

        return total_loss, details

    @staticmethod
    def _new_running_stats() -> Dict[str, float]:
        return {}

    @staticmethod
    def _accumulate_stats(
        running: Dict[str, float],
        details: Mapping[str, torch.Tensor],
    ) -> None:
        for key, value in details.items():
            if torch.is_tensor(value):
                scalar = float(value.detach().float().cpu())
            else:
                scalar = float(value)
            running[key] = running.get(key, 0.0) + scalar

    @staticmethod
    def _average_stats(
        running: Dict[str, float],
        count: int,
    ) -> Dict[str, float]:
        denominator = max(1, int(count))
        return {
            key: value / denominator
            for key, value in running.items()
        }

    # ==============================================================================================
    # Train / validation
    # ==============================================================================================

    def _train_step(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        running = self._new_running_stats()
        batch_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_dataloader):
            batch = self._move_batch_to_device(batch)

            with self.accelerator.accumulate(self.model):
                data = batch["image"]
                labels = batch["label"]

                predicted, aux_outputs = self.model(
                    data,
                    return_aux=True,
                )

                loss, details = self._compute_total_loss(
                    predicted=predicted,
                    labels=labels,
                    aux_outputs=aux_outputs,
                    epoch=epoch,
                )

                self.accelerator.backward(loss)

                if (
                    self.clip_gradients_enabled
                    and self.accelerator.sync_gradients
                ):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_gradients_value,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Update EMA only after a real synchronized optimizer step.
                if (
                    self.use_ema
                    and self.accelerator.sync_gradients
                ):
                    self._update_ema_model()

            self._accumulate_stats(running, details)
            batch_count += 1

            if (
                self.print_every > 0
                and (batch_idx + 1) % self.print_every == 0
                and self.accelerator.is_local_main_process
            ):
                print(
                    f"[train batch {batch_idx + 1:04d}] "
                    f"total={float(details['total']):.5f} "
                    f"seg={float(details['seg']):.5f} "
                    f"capf={float(details['capf_unscaled']):.5f} "
                    f"scale={float(details['capf_scale']):.5f}"
                )

        return self._average_stats(running, batch_count)

    @torch.no_grad()
    def _val_step(
        self,
        epoch: int,
        use_ema: bool = False,
    ):
        if use_ema and self.use_ema:
            self.ema_model.eval()
            model_ref = self.ema_model
        else:
            self.model.eval()
            model_ref = self.model

        running = self._new_running_stats()
        batch_count = 0

        if self.calculate_metrics:
            self.sliding_window_inferer.reset_metric()

        for batch in self.val_dataloader:
            batch = self._move_batch_to_device(batch)

            data = batch["image"]
            labels = batch["label"]

            predicted, aux_outputs = model_ref(
                data,
                return_aux=True,
            )

            _, details = self._compute_total_loss(
                predicted=predicted,
                labels=labels,
                aux_outputs=aux_outputs,
                epoch=epoch,
            )

            self._accumulate_stats(running, details)
            batch_count += 1

            if self.calculate_metrics:
                # SlidingWindowInference calls model_ref(data) without return_aux,
                # so the model returns segmentation logits only.
                self.sliding_window_inferer(
                    data,
                    labels,
                    model_ref,
                )

        avg_details = self._average_stats(running, batch_count)

        if self.calculate_metrics:
            metrics_dict = (
                self.sliding_window_inferer.aggregate_and_reset()
            )
        else:
            metrics_dict = {
                "val_mean_dice": 0.0,
                "TC": 0.0,
                "WT": 0.0,
                "ET": 0.0,
            }

        return avg_details, metrics_dict

    # ==============================================================================================
    # Checkpointing
    # ==============================================================================================

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        use_ema: bool = False,
    ) -> None:
        os.makedirs(self.checkpoint_save_dir, exist_ok=True)

        if use_ema and self.use_ema:
            model_to_save = self.ema_model.state_dict()
            prefix = "ema_"
        else:
            model_to_save = (
                self.accelerator.unwrap_model(self.model).state_dict()
            )
            prefix = ""

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model_to_save,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dice": float(self.best_dice),
            "capf_loss_config": self.capf_cfg,
        }

        if self.train_scheduler is not None:
            checkpoint["train_scheduler_state_dict"] = (
                self.train_scheduler.state_dict()
            )

        if self.warmup_scheduler is not None:
            checkpoint["warmup_scheduler_state_dict"] = (
                self.warmup_scheduler.state_dict()
            )

        latest_path = os.path.join(
            self.checkpoint_save_dir,
            f"{prefix}checkpoint_latest.pth",
        )
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(
                self.checkpoint_save_dir,
                f"{prefix}checkpoint_best.pth",
            )
            torch.save(checkpoint, best_path)

    # ==============================================================================================
    # Main loop
    # ==============================================================================================

    def train(self):
        warmup_epochs = int(
            self.config["warmup_scheduler"]["warmup_epochs"]
        )

        epoch_bar = tqdm(
            range(self.num_epochs),
            total=self.num_epochs,
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in epoch_bar:
            self.current_epoch = epoch

            if self.accelerator.is_local_main_process:
                if (
                    epoch == 0
                    and self.warmup_scheduler is not None
                    and epoch < warmup_epochs
                ):
                    print("[info] -- warming up learning rate")
                elif (
                    epoch == warmup_epochs
                    and self.train_scheduler is not None
                ):
                    print(
                        "[info] -- switching to learning rate decay schedule"
                    )

            train_details = self._train_step(epoch)

            if (
                self.warmup_scheduler is not None
                and epoch < warmup_epochs
            ):
                self.warmup_scheduler.step()
            elif self.train_scheduler is not None:
                self.train_scheduler.step()

            current_lr = float(self.optimizer.param_groups[0]["lr"])

            if self.accelerator.is_local_main_process:
                print(
                    f"epoch: {epoch:04d} -- "
                    f"train loss: {train_details.get('total', 0.0):.5f} -- "
                    f"lr: {current_lr}"
                )

            val_details, val_metrics = self._val_step(
                epoch=epoch,
                use_ema=False,
            )

            mean_dice = float(val_metrics["val_mean_dice"])

            save_flag = ""
            if mean_dice > self.best_dice:
                self.best_dice = mean_dice
                self._save_checkpoint(
                    epoch,
                    is_best=True,
                    use_ema=False,
                )
                save_flag = " -- saved"
            else:
                self._save_checkpoint(
                    epoch,
                    is_best=False,
                    use_ema=False,
                )

            if self.accelerator.is_local_main_process:
                # Keep the project's original parser-compatible main log line.
                print(
                    f"epoch -- {epoch:04d} || "
                    f"train loss -- {train_details.get('total', 0.0):.5f} || "
                    f"val loss -- {val_details.get('total', 0.0):.5f} || "
                    f"lr -- {current_lr:.8f} || "
                    f"val mean_dice -- {mean_dice:.5f} || "
                    f"TC -- {float(val_metrics['TC']):.5f} || "
                    f"WT -- {float(val_metrics['WT']):.5f} || "
                    f"ET -- {float(val_metrics['ET']):.5f}"
                    f"{save_flag}"
                )

                # CAPF-specific diagnostics are printed separately so existing
                # MATLAB / regex parsers for the main line remain valid.
                print(
                    f"[CAPF] epoch -- {epoch:04d} || "
                    f"scale -- {train_details.get('capf_scale', 0.0):.5f} || "
                    f"seg -- {train_details.get('seg', 0.0):.5f} || "
                    f"aux -- {train_details.get('capf_unscaled', 0.0):.5f} || "
                    f"primitive -- "
                    f"{train_details.get('primitive_supervision', 0.0):.5f} || "
                    f"evidence -- "
                    f"{train_details.get('evidence_supervision', 0.0):.5f} || "
                    f"boundary -- "
                    f"{train_details.get('boundary_loss', 0.0):.5f} || "
                    f"uncertainty -- "
                    f"{train_details.get('uncertainty_alignment', 0.0):.5f} || "
                    f"hierarchy -- "
                    f"{train_details.get('hierarchy_violation', 0.0):.6f}"
                )

                epoch_bar.set_postfix(
                    train_loss=f"{train_details.get('total', 0.0):.5f}",
                    val_loss=f"{val_details.get('total', 0.0):.5f}",
                    dice=f"{mean_dice:.5f}",
                    capf=f"{train_details.get('capf_scale', 0.0):.3f}",
                )

            if (
                self.use_ema
                and (epoch + 1) % self.val_ema_every == 0
            ):
                val_details_ema, val_metrics_ema = self._val_step(
                    epoch=epoch,
                    use_ema=True,
                )

                if self.accelerator.is_local_main_process:
                    print(
                        f"[EMA] epoch -- {epoch:04d} || "
                        f"val loss -- "
                        f"{val_details_ema.get('total', 0.0):.5f} || "
                        f"val mean_dice -- "
                        f"{float(val_metrics_ema['val_mean_dice']):.5f} || "
                        f"TC -- {float(val_metrics_ema['TC']):.5f} || "
                        f"WT -- {float(val_metrics_ema['WT']):.5f} || "
                        f"ET -- {float(val_metrics_ema['ET']):.5f}"
                    )

