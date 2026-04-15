import os
import glob
from typing import List
from src.utils.dataset import MyDataset

import pandas as pd
import argparse
import numpy as np
import logging
import csv
from typing import Optional, Dict, Iterable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_image_list(index_dir: str = "./web400_cleanlab_index") -> List[str]:
    """
    Traverse index_dir for all .csv files, load the column named 'image' (fall back to the second column),
    and return a concatenated list of values (as strings).
    """
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"{index_dir} does not exist or is not a directory")

    images: List[str] = []
    pattern = os.path.join(index_dir, "*.csv")
    for path in sorted(glob.glob(pattern)):
        try:
            # prefer named column 'image'
            df = pd.read_csv(path, usecols=["image"])
            col = df["image"].astype(str).tolist()
        except Exception:
            # fallback: take second column (index 1) if available
            df = pd.read_csv(path, header=0)
            if df.shape[1] < 2:
                # skip files without a second column
                continue
            col = df.iloc[:, 1].astype(str).tolist()

        images.extend(col)

    return images


def _try_instantiate_dataset(dataset_root: Optional[str] = None) -> Optional[MyDataset]:
    """
    Try to create a MyDataset instance with several common signatures.
    Return None if instantiation fails.
    """
    return MyDataset(root_dir=dataset_root, num_class=5000) if dataset_root else None


def _extract_paths_from_dataset(dataset: MyDataset) -> List[str]:
    """
    Extract a list of image path strings from a MyDataset instance by probing
    common attributes or iterating the dataset once.
    """
    return dataset.dataset_imgs


def _scan_filesystem_for_images(root: str, exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".gif")) -> List[str]:
    """
    Recursively scan root and return sorted list of image file paths.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def _build_key_to_index_map(paths: List[str]) -> Dict[str, int]:
    """
    Create mapping keys -> index to ease matching:
    - normalized full path
    - basename
    - basename without extension
    """
    m = {}
    for idx, p in enumerate(paths):
        m[p] = idx
    return m


def find_indices_for_images(image_names: List[str], dataset_root: Optional[str] = None, output: str = "clean_indices.npy") -> List[int]:
    """
    Main helper: attempt to get dataset image paths, build mapping, and match image_names -> indices.
    Returns list of indices (in dataset order) and writes numpy file.
    """
    # Try to instantiate dataset
    dset = _try_instantiate_dataset(dataset_root)
    if dset is not None:
        logging.info("Instantiated MyDataset successfully; extracting image paths from dataset.")
        paths = _extract_paths_from_dataset(dset)
    else:
        if dataset_root is None:
            raise RuntimeError("Unable to instantiate MyDataset and no dataset_root provided. Provide --dataset-root to fall back to filesystem scanning.")
        logging.info("Could not instantiate MyDataset; falling back to filesystem scan of dataset_root.")
        paths = _scan_filesystem_for_images(dataset_root)

    if not paths:
        raise RuntimeError("No image paths extracted from dataset or filesystem scan. Check dataset or dataset_root.")

    key_map = _build_key_to_index_map(paths)

    indices = []
    for name in image_names:
        indices.append(key_map[name])
        continue

    indices_arr = np.array(indices, dtype=np.int64)
    np.save(output, indices_arr)
    logging.info("Saved %d indices to %s", len(indices_arr), output)
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load image list from index CSVs and map to MyDataset indices; save to npy.")
    parser.add_argument("--index_dir", type=str, default="output/web5000_cleanlab_index/indices", help="Directory containing CSV index files.")
    parser.add_argument("--dataset_root", type=str, default='/mnt/7T/xz/wjl/webinat5000_train/train', help="Root of the dataset (used if MyDataset cannot be instantiated).")
    parser.add_argument("--output", type=str, default="output/web5000_cleanlab_index/indices/clean_indices.npy", help="Output .npy file to save indices.")
    args = parser.parse_args()

    imgs = load_image_list(args.index_dir)
    # 将imgs保存到csv文件
    with open(args.index_dir+'/clean_image_paths.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image'])
        for img in imgs:
            writer.writerow([img])
    logging.info("Loaded %d image entries from %s", len(imgs), args.index_dir)
    find_indices_for_images(imgs, dataset_root=args.dataset_root, output=args.output)

