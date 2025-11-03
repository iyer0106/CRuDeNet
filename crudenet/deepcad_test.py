import os
import glob
from typing import Optional, Dict, Tuple
from .deepcad.test_collection import testing_class


def test_deepcad(
    datasets_path: str,
    denoise_model: str,
    pth_dir: str = "./pth",
    output_dir: str = "./results",
    fmap: int = 16,
    GPU: str = "0",
    patch_xy: int = 150,
    patch_t: int = 8,
    overlap_factor: float = 0.4,
    test_datasize: int = 500,
    scale_factor: int = 1,
    num_workers: int = 0,
    visualize_images_per_epoch: bool = False,
    save_test_images_per_epoch: bool = True,
) -> Tuple[Dict, str]:
    """Test DeepCAD-RT model and return (config, output_path).

    Returns the output path where denoised results are saved.
    """
    test_dict = {
        "patch_x": patch_xy,
        "patch_y": patch_xy,
        "patch_t": patch_t,
        "overlap_factor": overlap_factor,
        "scale_factor": scale_factor,
        "test_datasize": test_datasize,
        "datasets_path": datasets_path,
        "pth_dir": pth_dir,
        "denoise_model": denoise_model,
        "output_dir": output_dir,
        "fmap": fmap,
        "GPU": GPU,
        "num_workers": num_workers,
        "visualize_images_per_epoch": visualize_images_per_epoch,
        "save_test_images_per_epoch": save_test_images_per_epoch,
    }
    tc = testing_class(test_dict)
    tc.run()
    return test_dict, tc.output_path


def get_deepcad_output_path(
    datasets_path: str,
    denoise_model: str,
    output_dir: str = "./results",
    epoch: Optional[int] = None,
    iteration: Optional[int] = None,
) -> str:
    """Get the expected output path for DeepCAD denoised results.
    
    DeepCAD saves results in a nested structure like:
    results/DataFolderIs_{dataset}_*/ModelFolderIs_{model}_*/E_{epoch}_Iter_{iter}/*_output.tif
    """
    # Find the most recent result folder
    pattern = os.path.join(output_dir, f"*ModelFolderIs_{denoise_model}*")
    model_dirs = glob.glob(pattern)
    if not model_dirs:
        raise FileNotFoundError(f"No output directory found matching {pattern}")
    
    model_dir = sorted(model_dirs)[-1]
    
    if epoch is not None and iteration is not None:
        epoch_dir = os.path.join(model_dir, f"E_{epoch}_Iter_{iteration}")
        if os.path.exists(epoch_dir):
            output_files = glob.glob(os.path.join(epoch_dir, "*_output.tif"))
            if output_files:
                return output_files[0]
    
    # Find the latest epoch/iteration folder
    epoch_dirs = glob.glob(os.path.join(model_dir, "E_*_Iter_*"))
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {model_dir}")
    
    latest_dir = sorted(epoch_dirs, key=lambda x: os.path.getmtime(x))[-1]
    output_files = glob.glob(os.path.join(latest_dir, "*_output.tif"))
    if output_files:
        return output_files[0]
    
    raise FileNotFoundError(f"No output .tif file found in {latest_dir}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser("DeepCAD-RT testing via CRuDeNet")
    p.add_argument("--datasets-path", required=True, help="Folder containing .tif files to denoise")
    p.add_argument("--denoise-model", required=True, help="Model folder name (e.g., 'BrainSlice')")
    p.add_argument("--pth-dir", default="./pth", help="Directory containing model checkpoints")
    p.add_argument("--output-dir", default="./results", help="Output directory for results")
    p.add_argument("--fmap", type=int, default=16)
    p.add_argument("--gpu", default="0", help="GPU index")
    p.add_argument("--patch-xy", type=int, default=150)
    p.add_argument("--patch-t", type=int, default=8)
    p.add_argument("--overlap-factor", type=float, default=0.4)
    p.add_argument("--test-datasize", type=int, default=500)
    p.add_argument("--scale-factor", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--visualize-images-per-epoch", action="store_true")
    p.add_argument("--save-test-images-per-epoch", action="store_true", default=True)
    args = p.parse_args()

    config, output_path = test_deepcad(
        datasets_path=args.datasets_path,
        denoise_model=args.denoise_model,
        pth_dir=args.pth_dir,
        output_dir=args.output_dir,
        fmap=args.fmap,
        GPU=args.gpu,
        patch_xy=args.patch_xy,
        patch_t=args.patch_t,
        overlap_factor=args.overlap_factor,
        test_datasize=args.test_datasize,
        scale_factor=args.scale_factor,
        num_workers=args.num_workers,
        visualize_images_per_epoch=args.visualize_images_per_epoch,
        save_test_images_per_epoch=args.save_test_images_per_epoch,
    )
    print(f"\nResults saved to: {output_path}")
    
    # Try to get the specific output file
    try:
        output_file = get_deepcad_output_path(
            args.datasets_path, args.denoise_model, args.output_dir
        )
        print(f"Output file: {output_file}")
    except Exception as e:
        print(f"Could not auto-detect output file: {e}")


if __name__ == "__main__":
    main()

