import argparse
import math
import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brats2017"))

from architectures.build_architecture import build_architecture


ORGAN_NAMES = [
    "Aorta",
    "Gallbladder",
    "Left Kidney",
    "Right Kidney",
    "Liver",
    "Pancreas",
    "Spleen",
    "Stomach",
]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes=9):
    config = {
        "model_name": "SREFormer",
        "model_parameters": {
            "in_channels": 1,
            "num_classes": num_classes,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [32, 64, 160, 256],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [4, 4, 4, 4],
            "num_heads": [1, 2, 5, 8],
            "depths": [2, 2, 2, 2],
            "decoder_head_embedding_dim": 256,
            "decoder_dropout": 0.0,
        },
    }
    return build_architecture(config)


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=9, include_background=False, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.long(), num_classes=self.num_classes)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()

        if not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_oh, dims)
        denom = torch.sum(probs + target_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def compute_lr(global_step, total_steps, warmup_steps, warmup_start_lr, max_lr, power):
    if global_step < warmup_steps:
        alpha = global_step / max(1, warmup_steps)
        return warmup_start_lr + alpha * (max_lr - warmup_start_lr)
    t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * ((1.0 - t) ** power)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def print_cuda_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{prefix}cuda memory | allocated={allocated:.2f} GB | reserved={reserved:.2f} GB | max_allocated={max_allocated:.2f} GB")


class SynapsePatchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        patch_size=(64, 128, 128),
        patches_per_epoch=250,
        fg_prob=0.70,
        fg_sample_per_case=20000,
        seed=42,
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No npz found in {self.data_dir}")

        self.patch_size = tuple(int(x) for x in patch_size)
        self.patches_per_epoch = int(patches_per_epoch)
        self.fg_prob = float(fg_prob)
        self.rng = np.random.default_rng(seed)

        self.cache = {}
        self.fg_samples = {}

        print("=> building foreground coordinate samples...")
        for f in self.files:
            arr = np.load(f)
            label = arr["label"].astype(np.int16)
            flat = np.flatnonzero(label > 0)
            if flat.size > 0:
                take = min(int(fg_sample_per_case), flat.size)
                chosen = self.rng.choice(flat, size=take, replace=False)
                coords = np.stack(np.unravel_index(chosen, label.shape), axis=1).astype(np.int32)
            else:
                coords = np.zeros((0, 3), dtype=np.int32)
            self.fg_samples[f.stem] = coords
            print(f"  {f.stem}: shape={label.shape}, sampled_fg={len(coords)}")

    def __len__(self):
        return self.patches_per_epoch

    def _load(self, file):
        key = file.stem
        if key not in self.cache:
            arr = np.load(file)
            image = arr["image"].astype(np.float32)
            label = arr["label"].astype(np.int16)
            self.cache[key] = (image, label)
        return self.cache[key]

    @staticmethod
    def _crop_with_pad(image, label, start, patch_size):
        pd, ph, pw = patch_size
        D, H, W = image.shape
        z0, y0, x0 = start
        z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw

        pad_before = [max(0, -z0), max(0, -y0), max(0, -x0)]
        pad_after = [max(0, z1 - D), max(0, y1 - H), max(0, x1 - W)]

        z0c, y0c, x0c = max(0, z0), max(0, y0), max(0, x0)
        z1c, y1c, x1c = min(D, z1), min(H, y1), min(W, x1)

        img = image[z0c:z1c, y0c:y1c, x0c:x1c]
        lab = label[z0c:z1c, y0c:y1c, x0c:x1c]

        if any(pad_before) or any(pad_after):
            pad_width = tuple((pad_before[i], pad_after[i]) for i in range(3))
            img = np.pad(img, pad_width, mode="constant", constant_values=0)
            lab = np.pad(lab, pad_width, mode="constant", constant_values=0)

        return img, lab

    def __getitem__(self, idx):
        file = random.choice(self.files)
        case = file.stem
        image, label = self._load(file)
        D, H, W = image.shape
        pd, ph, pw = self.patch_size

        use_fg = (random.random() < self.fg_prob) and (len(self.fg_samples[case]) > 0)
        if use_fg:
            center = self.fg_samples[case][random.randrange(len(self.fg_samples[case]))]
            zc, yc, xc = [int(v) for v in center]
            z0 = zc - pd // 2 + random.randint(-pd // 8, pd // 8)
            y0 = yc - ph // 2 + random.randint(-ph // 8, ph // 8)
            x0 = xc - pw // 2 + random.randint(-pw // 8, pw // 8)
        else:
            z0 = random.randint(0, max(0, D - pd)) if D > pd else 0
            y0 = random.randint(0, max(0, H - ph)) if H > ph else 0
            x0 = random.randint(0, max(0, W - pw)) if W > pw else 0

        # keep start mostly inside; padding handles edge cases
        z0 = min(max(z0, -pd // 2), max(0, D - pd))
        y0 = min(max(y0, -ph // 2), max(0, H - ph))
        x0 = min(max(x0, -pw // 2), max(0, W - pw))

        img, lab = self._crop_with_pad(image, label, (z0, y0, x0), self.patch_size)
        img = torch.from_numpy(img[None].astype(np.float32))   # 1,D,H,W
        lab = torch.from_numpy(lab.astype(np.int64))           # D,H,W
        return img, lab, case


def dice_per_class(pred, gt, num_classes=9):
    scores = []
    for c in range(1, num_classes):
        p = pred == c
        g = gt == c
        denom = p.sum() + g.sum()
        if denom == 0:
            scores.append(np.nan)
        else:
            scores.append((2.0 * np.logical_and(p, g).sum() / denom) * 100.0)
    return scores


def make_starts(length, roi, stride):
    if length <= roi:
        return [0]
    starts = list(range(0, length - roi + 1, stride))
    if starts[-1] != length - roi:
        starts.append(length - roi)
    return starts


def sliding_window_predict(
    model,
    image,
    device,
    roi_size=(64, 128, 128),
    overlap=0.5,
    sw_batch_size=2,
    num_classes=9,
    amp=True,
):
    model.eval()
    D, H, W = image.shape
    rd, rh, rw = roi_size
    sd, sh, sw = [max(1, int(r * (1.0 - overlap))) for r in roi_size]

    zs = make_starts(D, rd, sd)
    ys = make_starts(H, rh, sh)
    xs = make_starts(W, rw, sw)

    score = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    windows = []
    locs = []

    def flush():
        nonlocal windows, locs, score, count
        if not windows:
            return
        batch = torch.from_numpy(np.stack(windows, axis=0)[:, None]).float().to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = model(batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=1).float().cpu().numpy()

        for prob, (z, y, x, zd, yh, xw) in zip(probs, locs):
            score[:, z:z+zd, y:y+yh, x:x+xw] += prob[:, :zd, :yh, :xw]
            count[z:z+zd, y:y+yh, x:x+xw] += 1.0

        windows = []
        locs = []

    for z in zs:
        for y in ys:
            for x in xs:
                patch = image[z:z+rd, y:y+rh, x:x+rw]
                zd, yh, xw = patch.shape
                if patch.shape != roi_size:
                    pad = ((0, rd - zd), (0, rh - yh), (0, rw - xw))
                    patch = np.pad(patch, pad, mode="constant", constant_values=0)
                windows.append(patch.astype(np.float32))
                locs.append((z, y, x, zd, yh, xw))
                if len(windows) >= sw_batch_size:
                    flush()
    flush()

    score /= np.maximum(count[None], 1e-6)
    pred = np.argmax(score, axis=0).astype(np.uint8)
    return pred


def validate(model, val_dir, device, args):
    val_files = sorted(Path(val_dir).glob("*.npz"))
    all_scores = []
    per_case_means = []

    for f in val_files:
        arr = np.load(f)
        image = arr["image"].astype(np.float32)
        label = arr["label"].astype(np.int16)

        pred = sliding_window_predict(
            model,
            image,
            device,
            roi_size=tuple(args.patch_size),
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            num_classes=args.num_classes,
            amp=args.amp,
        )
        scores = dice_per_class(pred, label, num_classes=args.num_classes)
        all_scores.append(scores)
        mean_case = float(np.nanmean(scores))
        per_case_means.append(mean_case)
        print(f"{f.stem} | mean dice: {mean_case:.2f}")

    all_scores = np.array(all_scores, dtype=np.float32)
    per_class = np.nanmean(all_scores, axis=0)
    mean_dice = float(np.nanmean(per_class))

    print("val mean_dice:", f"{mean_dice:.2f}")
    print("per-class dice:", [round(float(x), 2) for x in per_class])
    print("per-class names:", ORGAN_NAMES)

    return mean_dice, per_class


def save_checkpoint(path, model, optimizer, epoch, best_dice, args):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dice": best_dice,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder with train/ and val/ original volume npz.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128])
    parser.add_argument("--patches_per_epoch", type=int, default=250)
    parser.add_argument("--fg_prob", type=float, default=0.70)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--warmup_start_lr", type=float, default=4e-6)
    parser.add_argument("--warmup_end_lr", type=float, default=4e-4)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--debug_batch", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=> device:", device)
    print("=> data_dir:", data_dir)
    print("=> out_dir:", out_dir)
    print("=> protocol: 8-organ original-volume patch training + sliding-window validation")
    print("=> config source: INTERNAL Python dict in build_model(); external config.yaml is NOT used")
    print("=> epochs:", args.epochs)
    print("=> batch_size:", args.batch_size)
    print("=> patch_size:", args.patch_size)
    print("=> patches_per_epoch:", args.patches_per_epoch)
    print("=> fg_prob:", args.fg_prob)
    print("=> optimizer: AdamW")
    print("=> loss: 0.5 CrossEntropy + 0.5 foreground Dice")
    print("=> scheduler: warmup 4e-6 -> 4e-4 + PolyLR")
    print("=> val: sliding window in original D x 512 x 512 space")
    print("=> amp:", args.amp)

    train_set = SynapsePatchDataset(
        train_dir,
        patch_size=tuple(args.patch_size),
        patches_per_epoch=args.patches_per_epoch,
        fg_prob=args.fg_prob,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )

    print(f"=> train volumes: {len(list(train_dir.glob('*.npz')))}")
    print(f"=> val volumes: {len(list(val_dir.glob('*.npz')))}")
    print(f"=> REAL train_loader batch_size: {train_loader.batch_size}")
    print(f"=> REAL train_loader steps_per_epoch: {len(train_loader)}")

    model = build_model(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.warmup_start_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = SoftDiceLoss(num_classes=args.num_classes, include_background=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"=> model params: {total_params/1e6:.3f} M")

    start_epoch = 1
    best_dice = -1.0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", -1.0))
        print(f"=> resumed from {args.resume}")
        print(f"=> start_epoch={start_epoch}, best_dice={best_dice:.2f}")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = []
        t0 = time.time()

        for step, (image, label, case) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()

            if args.debug_batch and epoch == start_epoch and step == 1:
                print("=> FIRST TRAIN BATCH image shape:", tuple(image.shape))
                print("=> FIRST TRAIN BATCH label shape:", tuple(label.shape))
                print("=> FIRST TRAIN BATCH cases:", case)
                print_cuda_memory("=> before first forward | ")

            lr_now = compute_lr(
                global_step,
                total_steps,
                warmup_steps,
                args.warmup_start_lr,
                args.warmup_end_lr,
                args.poly_power,
            )
            set_lr(optimizer, lr_now)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                logits = model(image)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss_ce = ce_loss(logits, label)
                loss_dice = dice_loss(logits, label)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running.append(float(loss.detach().cpu()))

            if args.debug_batch and epoch == start_epoch and step == 1:
                print_cuda_memory("=> after first backward/update | ")

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch:04d} | step {step:04d}/{len(train_loader)} | "
                    f"lr {lr_now:.8f} | loss {float(loss.detach()):.4f} | "
                    f"ce {float(loss_ce.detach()):.4f} | dice {float(loss_dice.detach()):.4f}"
                )

        print(f"epoch {epoch:04d} finished | train_loss {float(np.mean(running)):.4f} | time {time.time() - t0:.1f}s")

        latest_path = out_dir / "checkpoint_latest.pth"
        save_checkpoint(latest_path, model, optimizer, epoch, best_dice, args)

        if epoch % args.val_interval == 0 or epoch == 1 or epoch == args.epochs:
            mean_dice, per_class = validate(model, val_dir, device, args)

            if mean_dice > best_dice:
                best_dice = mean_dice
                best_path = out_dir / "checkpoint_best.pth"
                save_checkpoint(best_path, model, optimizer, epoch, best_dice, args)
                print(f"=> saved best checkpoint, best_epoch={epoch}, best_dice={best_dice:.2f}")

    print("training finished.")
    print("best_dice:", f"{best_dice:.2f}")


if __name__ == "__main__":
    main()
