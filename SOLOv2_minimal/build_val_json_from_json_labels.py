import os
import json
import cv2

# ================= CONFIG =================
DATA_ROOT = "datasets/vid3_nori"
IMG_DIR = os.path.join(DATA_ROOT, "images/val")
ANN_DIR = os.path.join(DATA_ROOT, "labels/val")

OUT_DIR = "results/val"
OUT_JSON = os.path.join(OUT_DIR, "custom_annotations.json")

CLASS_NAMES = [
    "1_G", "1_NG", "2_G", "2_NG",
    "3_G", "3_NG", "4_G", "4_NG",
    "5_G", "5_NG"
]
# =========================================

os.makedirs(OUT_DIR, exist_ok=True)

images = []
annotations = []
categories = []

class2id = {name: i for i, name in enumerate(CLASS_NAMES)}

# categories
for name, idx in class2id.items():
    categories.append({
        "id": idx,
        "name": name,
        "supercategory": "object"
    })

ann_id = 1
img_id = 1

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    ann_path = os.path.join(
        ANN_DIR, os.path.splitext(img_name)[0] + ".json"
    )

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    images.append({
        "id": img_id,
        "file_name": img_name,
        "height": h,
        "width": w
    })

    if not os.path.exists(ann_path):
        img_id += 1
        continue

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for poly in data.get("polygons", []):
        if poly.get("shape_type") != "Polygon":
            continue

        class_name = poly.get("category")
        if class_name not in class2id:
            continue

        pts = poly.get("img_points", [])
        if len(pts) < 3:
            continue

        # flatten [[x,y],...] -> [x1,y1,x2,y2,...]
        segmentation = []
        xs, ys = [], []
        for x, y in pts:
            segmentation.extend([float(x), float(y)])
            xs.append(x)
            ys.append(y)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bbox = [
            float(x_min),
            float(y_min),
            float(x_max - x_min),
            float(y_max - y_min)
        ]

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": class2id[class_name],
            "segmentation": [segmentation],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        ann_id += 1

    img_id += 1

img_ids = [img["id"] for img in images]
cate_ids = [cat["id"] for cat in categories]
class_names = [
    "1_G", "1_NG", "2_G", "2_NG",
    "3_G", "3_NG", "4_G", "4_NG",
    "5_G", "5_NG"
]
coco = {
    "images": images,
    "annotations": annotations,
    "categories": categories,
    "img_ids": img_ids,
    "cate_ids": cate_ids,
    "class_names": class_names
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2)

print(f"[OK] Saved: {OUT_JSON}")
print(f"Images: {len(images)} | Annotations: {len(annotations)}")
