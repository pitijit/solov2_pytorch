import json
import os
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO


def check_coco_gt(json_path, img_dir):
    print(f"\n===== Checking GT: {json_path} =====\n")

    coco = COCO(json_path)

    # -----------------------------
    # Basic stats
    # -----------------------------
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds()

    print(f"Total images      : {len(img_ids)}")
    print(f"Total categories  : {len(cat_ids)}")
    print(f"Total annotations : {len(ann_ids)}\n")

    # -----------------------------
    # Category id check
    # -----------------------------
    print("== Category ID Check ==")
    cat_ids_sorted = sorted(cat_ids)
    expected = list(range(1, len(cat_ids_sorted) + 1))

    if cat_ids_sorted != expected:
        print("❌ Category IDs are NOT continuous!")
        print("Found   :", cat_ids_sorted)
        print("Expected:", expected)
    else:
        print("✅ Category IDs are continuous")

    # -----------------------------
    # Image without GT
    # -----------------------------
    print("\n== Images without GT ==")
    empty_imgs = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            empty_imgs.append(img_id)

    print(f"Images without GT: {len(empty_imgs)}")
    if empty_imgs:
        print("Sample empty image IDs:", empty_imgs[:10])

    # -----------------------------
    # Annotation integrity check
    # -----------------------------
    print("\n== Annotation Integrity Check ==")
    bad_anns = defaultdict(list)

    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns(ann_id)[0]

        # bbox check
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            bad_anns["invalid_bbox"].append(ann_id)

        # area check
        if ann.get("area", 0) <= 0:
            bad_anns["zero_area"].append(ann_id)

        # category id check
        if ann["category_id"] not in cat_ids:
            bad_anns["invalid_category"].append(ann_id)

        # segmentation check
        seg = ann.get("segmentation", [])
        if not seg or len(seg) == 0:
            bad_anns["empty_segmentation"].append(ann_id)
        else:
            # polygon check
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) < 6:
                        bad_anns["invalid_polygon"].append(ann_id)
                        break

    for k, v in bad_anns.items():
        print(f"{k:20s}: {len(v)}")

    # -----------------------------
    # Image file existence
    # -----------------------------
    print("\n== Image File Check ==")
    missing_files = []
    for img in coco.loadImgs(img_ids):
        img_path = os.path.join(img_dir, img["file_name"])
        if not os.path.exists(img_path):
            missing_files.append(img["file_name"])

    print(f"Missing image files: {len(missing_files)}")
    if missing_files:
        print("Sample missing files:", missing_files[:10])

    print("\n===== DONE =====\n")


if __name__ == "__main__":
    DATA_ROOT = "datasets/vid3_nori"

    check_coco_gt(
        json_path=os.path.join(DATA_ROOT, "labels/train/"),
        img_dir=os.path.join(DATA_ROOT, "images/train"),
    )

    check_coco_gt(
        json_path=os.path.join(DATA_ROOT, "labels/val/"),
        img_dir=os.path.join(DATA_ROOT, "images/val"),
    )
