import os
import json
from collections import defaultdict
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
COCO_JSON = 'datasets/vid3_nori/labels/val/val.json'
IMAGE_DIR = 'datasets/vid3_nori/images/val'
OUT_LABEL_DIR = 'datasets/vid3_nori/labels/val'
USE_IMAGE_FILENAME = True  # ใช้ชื่อไฟล์ภาพเป็นชื่อ JSON

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# ==========================
# LOAD COCO JSON
# ==========================
with open(COCO_JSON, 'r', encoding='utf-8') as f:
    coco = json.load(f)

images = {img['id']: img for img in coco['images']}
categories = {cat['id']: cat['name'] for cat in coco['categories']}

# Group annotations by image_id
ann_by_image = defaultdict(list)
for ann in coco['annotations']:
    ann_by_image[ann['image_id']].append(ann)

# ==========================
# EXPORT PER-IMAGE LABELME JSON
# ==========================
for image_id, img_info in tqdm(images.items(), desc='Exporting'):
    anns = ann_by_image.get(image_id, [])

    if USE_IMAGE_FILENAME:
        img_name = os.path.splitext(img_info['file_name'])[0]
    else:
        img_name = str(image_id)

    out_json = {
        'polygons': [],
        'img_height': img_info.get('height'),
        'img_width': img_info.get('width'),
        'categories': list(categories.values())
    }

    for ann in anns:
        shape_type = 'Polygon'

        # แปลง segmentation COCO → points
        if isinstance(ann['segmentation'], list):
            if len(ann['segmentation']) == 1:
                points_flat = ann['segmentation'][0]
                img_points = [[points_flat[i], points_flat[i+1]] for i in range(0, len(points_flat), 2)]
            else:
                img_points = []
                for poly in ann['segmentation']:
                    poly_points = [[poly[i], poly[i+1]] for i in range(0, len(poly), 2)]
                    img_points.append(poly_points)
        else:
            raise ValueError(f"Unknown segmentation format for ann id {ann['id']}")

        out_json['polygons'].append({
            'shape_type': shape_type,
            'category': categories[ann['category_id']],
            'img_points': img_points
        })

    save_path = os.path.join(OUT_LABEL_DIR, f'{img_name}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

print('✅ Done: COCO JSON split into LabelMe per-image JSON completed.')
