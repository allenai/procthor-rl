import argparse
import os
from multiprocessing import Pool

from objathor.utils.download_utils import download_with_progress_bar

ALL_CKPT_IDS = {
    "CLIP-GRU": "exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt",
    "CLIP-CodeBook-GRU": "exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO-CodeBook__stage_02__steps_000420684456.pt",
    "DINOv2-GRU": "exp_ObjectNav-RGB-DINOv2GRU-DDPPO__stage_02__steps_000400477104.pt",
    "DINOv2-CodeBook-GRU": "exp_ObjectNav-RGB-DINOv2GRU-DDPPO-CodeBook__stage_02__steps_000405359832.pt",
    "DINOv2-ViTs-TSFM": "exp_ObjectNav-RGB-DINOv2-TSFM-TX-ENCODER-DDPPO__stage_02__steps_000437127636.pt",
    "DINOv2-ViTb-TSFM": "exp_ObjectNav-RGB-DINOv2-ViTb-TSFM-TX-ENCODER-DDPPO__stage_02__steps_000435703596.pt",
}


def download_ckpt(info):
    url = info["url"]
    save_dir = info["save_dir"]
    ckpt_id = info["ckpt_id"]

    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, ckpt_id)
    download_with_progress_bar(
        url=url,
        save_path=ckpt_path,
        desc=f"Downloading: checkpoint_final.ckpt.",
    )


def main():
    parser = argparse.ArgumentParser(description="Trained ckpt downloader.")
    parser.add_argument("--save_dir", required=True, help="Directory to save the downloaded files.")
    parser.add_argument(
        "--ckpt_ids",
        default=None,
        nargs="+",
        help=f"List of ckpt names to download, by default this will include all ids. Should be a subset of: {ALL_CKPT_IDS.keys()}",
    )
    parser.add_argument("--num", "-n", default=1, type=int, help="Number of parallel downloads.")
    args = parser.parse_args()

    if args.ckpt_ids is None:
        args.ckpt_ids = ALL_CKPT_IDS.keys()

    os.makedirs(args.save_dir, exist_ok=True)

    download_args = []
    for ckpt_id in args.ckpt_ids:
        ckpt = ALL_CKPT_IDS.get(ckpt_id)
        download_args.append(
            dict(
                url=f"https://pub-fbf23a0d54a0460882efdb338eb7282c.r2.dev/{ckpt}",
                save_dir=args.save_dir,
                ckpt_id=ckpt,
            )
        )

    with Pool(args.num) as pool:
        pool.map(download_ckpt, download_args)


if __name__ == "__main__":
    main()