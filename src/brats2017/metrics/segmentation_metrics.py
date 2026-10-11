import torch
import torch.nn as nn
from typing import Dict, Tuple
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference


class SlidingWindowInference:
    def __init__(self, roi: Tuple[int, int, int], sw_batch_size: int) -> None:
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean_batch",
            get_not_nans=False,
        )

        self.post_transform = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(argmax=False, threshold=0.5),
            ]
        )

        self.sw_batch_size = sw_batch_size
        self.roi = roi

    def reset_metric(self) -> None:
        self.dice_metric.reset()

    def _build_predictor(self, model: nn.Module):
        def predictor(x):
            out = model(x, return_aux=False)
            return out
        return predictor

    @torch.no_grad()
    def __call__(
        self,
        val_inputs: torch.Tensor,
        val_labels: torch.Tensor,
        model: nn.Module,
    ) -> None:
        predictor = self._build_predictor(model)

        logits = sliding_window_inference(
            inputs=val_inputs,
            roi_size=self.roi,
            sw_batch_size=self.sw_batch_size,
            predictor=predictor,
            overlap=0.5,
        )

        val_labels_list = decollate_batch(val_labels)
        val_outputs_list = decollate_batch(logits)

        val_output_convert = [
            self.post_transform(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]

        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)

    @torch.no_grad()
    def aggregate_and_reset(self) -> Dict[str, float]:
        dice = self.dice_metric.aggregate()
        self.dice_metric.reset()

        if isinstance(dice, torch.Tensor):
            dice = dice.detach().float().cpu().view(-1)
        else:
            dice = torch.as_tensor(dice, dtype=torch.float32).view(-1)

        if dice.numel() >= 3:
            tc = float(dice[0].item() * 100.0)
            wt = float(dice[1].item() * 100.0)
            et = float(dice[2].item() * 100.0)
            mean_dice = (tc + wt + et) / 3.0
        elif dice.numel() > 0:
            mean_dice = float(dice.mean().item() * 100.0)
            tc = wt = et = mean_dice
        else:
            mean_dice = tc = wt = et = 0.0

        return {
            "val_mean_dice": mean_dice,
            "TC": tc,
            "WT": wt,
            "ET": et,
        }
