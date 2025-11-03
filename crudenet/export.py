import os
from typing import Optional
import torch

from .model import build_model


def export_onnx(
    ckpt_path: Optional[str],
    out_dir: str,
    patch_t: int = 8,
    patch_xy: int = 128,
    c_hid: int = 32,
    num_layers: int = 3,
    opset: int = 13,
    device: Optional[str] = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model(1, c_hid, num_layers).to(device)

    if ckpt_path:
        net.load_state_dict(torch.load(ckpt_path, map_location=device))

    net.eval()
    dummy = torch.randn(1, 1, patch_t, patch_xy, patch_xy, device=device)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "crudenet_convgru_causal.onnx")
    torch.onnx.export(
        net,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX to {onnx_path}")
    return onnx_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("CRuDeNet ONNX export")
    p.add_argument("--ckpt", default=None, help="Optional checkpoint .pth to load")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--patch-t", type=int, default=8)
    p.add_argument("--patch-xy", type=int, default=128)
    p.add_argument("--c-hid", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    export_onnx(
        ckpt_path=args.ckpt,
        out_dir=args.out,
        patch_t=args.patch_t,
        patch_xy=args.patch_xy,
        c_hid=args.c_hid,
        num_layers=args.num_layers,
        opset=args.opset,
        device=args.device,
    )


if __name__ == "__main__":
    main()


