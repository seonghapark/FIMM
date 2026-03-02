import argparse
import json
import os
import random
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_subset(
    coco_root: Path,
    output_root: Path,
    train_count: int,
    test_count: int,
    seed: int,
):
    ann_path = coco_root / "annotations" / "instances_train2017.json"
    train_img_root = coco_root / "train2017"

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {ann_path}")
    if not train_img_root.exists():
        raise FileNotFoundError(f"Missing train image folder: {train_img_root}")

    with ann_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    licenses = data.get("licenses", [])
    info = data.get("info", {})

    total_needed = train_count + test_count
    if len(images) < total_needed:
        raise ValueError(
            f"Not enough images in train2017. Requested {total_needed}, found {len(images)}"
        )

    random.seed(seed)
    shuffled = images[:]
    random.shuffle(shuffled)

    train_images = shuffled[:train_count]
    test_images = shuffled[train_count:train_count + test_count]

    train_ids = {img["id"] for img in train_images}
    test_ids = {img["id"] for img in test_images}

    train_annotations = [ann for ann in annotations if ann.get("image_id") in train_ids]
    test_annotations = [ann for ann in annotations if ann.get("image_id") in test_ids]

    output_root.mkdir(parents=True, exist_ok=True)
    train_out_img = output_root / "train2017"
    test_out_img = output_root / "test2017"
    ann_out_dir = output_root / "annotations"
    ann_out_dir.mkdir(parents=True, exist_ok=True)

    for img in train_images:
        src = train_img_root / img["file_name"]
        dst = train_out_img / img["file_name"]
        link_or_copy(src, dst)

    for img in test_images:
        src = train_img_root / img["file_name"]
        dst = test_out_img / img["file_name"]
        link_or_copy(src, dst)

    train_json = {
        "info": info,
        "licenses": licenses,
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    test_json = {
        "info": info,
        "licenses": licenses,
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories,
    }

    with (ann_out_dir / "instances_train_subset_60k.json").open("w", encoding="utf-8") as file:
        json.dump(train_json, file)
    with (ann_out_dir / "instances_test_subset_10k.json").open("w", encoding="utf-8") as file:
        json.dump(test_json, file)

    manifest = {
        "source": str(coco_root),
        "seed": seed,
        "train_images": len(train_images),
        "train_annotations": len(train_annotations),
        "test_images": len(test_images),
        "test_annotations": len(test_annotations),
        "train_annotations_file": "annotations/instances_train_subset_60k.json",
        "test_annotations_file": "annotations/instances_test_subset_10k.json",
    }
    with (output_root / "subset_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("Subset generated successfully")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO subset with labeled train/test splits from train2017."
    )
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--output-root", type=Path, default=Path("data/coco/subset_60k_10k"))
    parser.add_argument("--train-count", type=int, default=60000)
    parser.add_argument("--test-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_subset(
        coco_root=args.coco_root,
        output_root=args.output_root,
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
