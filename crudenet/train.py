import os
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import build_model
from .dataset import VolumeDataset


def train(
    raw_path: str,
    gt_path: str,
    out_dir: str,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 2,
    patch_t: int = 8,
    patch_xy: int = 128,
    c_hid: int = 32,
    num_layers: int = 3,
    num_workers: int = 2,
    device: Optional[str] = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VolumeDataset(raw_path, gt_path, patch_t=patch_t, patch_xy=patch_xy)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    net = build_model(c_in=1, c_hid=c_hid, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        net.train()
        total = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {ep}"):
            x = x.to(device)  # [B,1,T,H,W]
            y = y.to(device)
            opt.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += float(loss.item())
        avg = total / max(1, len(loader))
        print(f"Epoch {ep}: loss {avg:.6f}")

    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "crudenet_convgru_causal.pth")
    torch.save(net.state_dict(), ckpt)
    print(f"Saved checkpoint to {ckpt}")
    return ckpt


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("CRuDeNet training")
    p.add_argument("--raw", required=True, help="Path to raw stack (.tif)")
    p.add_argument("--gt", required=True, help="Path to ground truth stack (.tif)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch-t", type=int, default=8)
    p.add_argument("--patch-xy", type=int, default=128)
    p.add_argument("--c-hid", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    train(
        raw_path=args.raw,
        gt_path=args.gt,
        out_dir=args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patch_t=args.patch_t,
        patch_xy=args.patch_xy,
        c_hid=args.c_hid,
        num_layers=args.num_layers,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()


