import os
import time
from typing import Optional
import numpy as np
import torch
import tifffile
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from .model import build_model
from .dataset import normalize01


def infer_stream(
    raw_path: str,
    ckpt_path: str,
    out_dir: str,
    gt_path: Optional[str] = None,
    c_hid: int = 32,
    num_layers: int = 3,
    device: Optional[str] = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    net = build_model(1, c_hid, num_layers).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    raw_stack = tifffile.imread(raw_path).astype(np.float32)
    raw_stack = normalize01(raw_stack)
    frames, H, W = raw_stack.shape

    gt_stack = None
    if gt_path and os.path.exists(gt_path):
        gt_stack = normalize01(tifffile.imread(gt_path).astype(np.float32))

    h_states = [None] * len(net.layers)
    outputs = []
    start = time.time()

    with torch.no_grad():
        for t in range(frames):
            x_t = torch.from_numpy(raw_stack[t:t+1][None, None]).float().to(device)
            for i, cell in enumerate(net.layers):
                h_states[i] = cell(x_t[:, 0], h_states[i])
                x_t = h_states[i].unsqueeze(1)
            y_t = net.out_conv(x_t[:, 0])
            outputs.append(y_t.cpu().numpy()[0, 0])

    runtime = time.time() - start
    fps = frames / max(runtime, 1e-9)
    mpxps = (frames * H * W / 1e6) / max(runtime, 1e-9)
    print(f"Frames: {frames} | Runtime: {runtime:.2f}s | {fps:.2f} fps | {mpxps:.2f} MPx/s")

    if gt_stack is not None:
        psnr_vals = [psnr(gt_stack[i], outputs[i], data_range=1.0) for i in range(frames)]
        ssim_vals = [ssim(gt_stack[i], outputs[i], data_range=1.0) for i in range(frames)]
        print(f"Mean PSNR: {np.mean(psnr_vals):.2f} dB, SSIM: {np.mean(ssim_vals):.3f}")

    denoised = (np.clip(np.stack(outputs), 0, 1) * 65535).astype(np.uint16)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "denoised_full_stack_causal.tif")
    tifffile.imwrite(save_path, denoised, photometric='minisblack')
    print(f"Saved: {save_path}")
    return save_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("CRuDeNet inference")
    p.add_argument("--raw", required=True, help="Path to raw stack (.tif)")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--gt", default=None, help="Optional GT stack for metrics")
    p.add_argument("--c-hid", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    infer_stream(
        raw_path=args.raw,
        ckpt_path=args.ckpt,
        out_dir=args.out,
        gt_path=args.gt,
        c_hid=args.c_hid,
        num_layers=args.num_layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()


