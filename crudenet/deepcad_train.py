import os
from typing import Optional, Dict
from .deepcad.train_collection import training_class


def train_deepcad(
    datasets_path: str,
    pth_dir: str = "./pth",
    n_epochs: int = 10,
    lr: float = 0.00005,
    b1: float = 0.5,
    b2: float = 0.999,
    fmap: int = 16,
    GPU: str = "0",
    patch_xy: int = 150,
    patch_t: int = 8,
    overlap_factor: float = 0.4,
    train_datasets_size: int = 2500,
    select_img_num: int = 1000000,
    scale_factor: int = 1,
    num_workers: int = 0,
    visualize_images_per_epoch: bool = False,
    save_test_images_per_epoch: bool = True,
) -> Dict:
    """Train DeepCAD-RT model and return training configuration."""
    train_dict = {
        "patch_x": patch_xy,
        "patch_y": patch_xy,
        "patch_t": patch_t,
        "overlap_factor": overlap_factor,
        "scale_factor": scale_factor,
        "select_img_num": select_img_num,
        "train_datasets_size": train_datasets_size,
        "datasets_path": datasets_path,
        "pth_dir": pth_dir,
        "n_epochs": n_epochs,
        "lr": lr,
        "b1": b1,
        "b2": b2,
        "fmap": fmap,
        "GPU": GPU,
        "num_workers": num_workers,
        "visualize_images_per_epoch": visualize_images_per_epoch,
        "save_test_images_per_epoch": save_test_images_per_epoch,
    }
    tc = training_class(train_dict)
    tc.run()
    return train_dict


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("DeepCAD-RT training via CRuDeNet")
    p.add_argument("--datasets-path", required=True, help="Folder containing .tif files for training")
    p.add_argument("--pth-dir", default="./pth", help="Output directory for checkpoints")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.00005)
    p.add_argument("--b1", type=float, default=0.5)
    p.add_argument("--b2", type=float, default=0.999)
    p.add_argument("--fmap", type=int, default=16)
    p.add_argument("--gpu", default="0", help="GPU index (e.g., '0' or '0,1')")
    p.add_argument("--patch-xy", type=int, default=150)
    p.add_argument("--patch-t", type=int, default=8)
    p.add_argument("--overlap-factor", type=float, default=0.4)
    p.add_argument("--train-datasets-size", type=int, default=2500)
    p.add_argument("--select-img-num", type=int, default=1000000)
    p.add_argument("--scale-factor", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0, help="Set to 0 on Windows")
    p.add_argument("--visualize-images-per-epoch", action="store_true")
    p.add_argument("--save-test-images-per-epoch", action="store_true", default=True)
    args = p.parse_args()

    train_deepcad(
        datasets_path=args.datasets_path,
        pth_dir=args.pth_dir,
        n_epochs=args.n_epochs,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
        fmap=args.fmap,
        GPU=args.gpu,
        patch_xy=args.patch_xy,
        patch_t=args.patch_t,
        overlap_factor=args.overlap_factor,
        train_datasets_size=args.train_datasets_size,
        select_img_num=args.select_img_num,
        scale_factor=args.scale_factor,
        num_workers=args.num_workers,
        visualize_images_per_epoch=args.visualize_images_per_epoch,
        save_test_images_per_epoch=args.save_test_images_per_epoch,
    )


if __name__ == "__main__":
    main()

