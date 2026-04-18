"""
Convert original NEU-DET dataset (Pascal VOC XML) to YOLO format.

Source: dataset_neudet_org/ (Kaggle)
  train/images/{class_name}/*.jpg
  train/annotations/*.xml
  validation/images/{class_name}/*.jpg
  validation/annotations/*.xml

Target: dataset_neudet_org_yolo/
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
  data.yaml
"""

import xml.etree.ElementTree as ET
import os
import glob
import shutil

# Paths
SRC = os.path.join(os.path.dirname(__file__), "dataset_neudet_org")
DST = os.path.join(os.path.dirname(__file__), "dataset_neudet_org_yolo")

# Class mapping (same order as Roboflow for consistency)
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

SPLIT_MAP = {
    "train": "train",
    "validation": "valid",
}


def convert_voc_to_yolo(xml_path):
    """Convert a single VOC XML annotation to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    filename = root.find("filename").text

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_TO_ID:
            print(f"  [WARN] Unknown class '{name}' in {xml_path}, skipping")
            continue

        cls_id = CLASS_TO_ID[name]
        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)

        # Clamp to image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        # Skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            print(f"  [WARN] Degenerate box in {xml_path}: ({xmin},{ymin},{xmax},{ymax})")
            continue

        # Convert to YOLO: cx cy bw bh (normalized)
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return filename, lines


def main():
    print("=" * 60)
    print("NEU-DET Original -> YOLO Format Converter")
    print("=" * 60)

    # Clean destination
    if os.path.exists(DST):
        shutil.rmtree(DST)

    stats = {}

    for src_split, dst_split in SPLIT_MAP.items():
        ann_dir = os.path.join(SRC, src_split, "annotations")
        img_base = os.path.join(SRC, src_split, "images")

        dst_img_dir = os.path.join(DST, dst_split, "images")
        dst_lbl_dir = os.path.join(DST, dst_split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        # Build image lookup: stem (no extension) -> (filename, full path)
        img_lookup = {}
        for cls_dir in glob.glob(os.path.join(img_base, "*")):
            if not os.path.isdir(cls_dir):
                continue
            for img_file in os.listdir(cls_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    stem = os.path.splitext(img_file)[0]
                    img_lookup[stem] = (img_file, os.path.join(cls_dir, img_file))

        # Process XML annotations
        xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        converted = 0
        total_objects = 0
        class_counts = {c: 0 for c in CLASSES}

        for xf in xml_files:
            filename, yolo_lines = convert_voc_to_yolo(xf)

            # Find corresponding image (XML filename may lack extension)
            stem = os.path.splitext(filename)[0]
            # If filename has no extension, stem == filename
            if stem not in img_lookup:
                print(f"  [WARN] Image not found for {filename}")
                continue

            img_file, img_full_path = img_lookup[stem]

            # Copy image
            dst_img_path = os.path.join(dst_img_dir, img_file)
            shutil.copy2(img_full_path, dst_img_path)

            # Write label
            dst_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
            with open(dst_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n")

            converted += 1
            total_objects += len(yolo_lines)
            for line in yolo_lines:
                cls_id = int(line.split()[0])
                class_counts[CLASSES[cls_id]] += 1

            # Remove from lookup so we can find unmatched images
            del img_lookup[stem]

        # Copy images without XML annotations (write empty label)
        orphan_count = 0
        for stem, (img_file, img_path) in img_lookup.items():
            shutil.copy2(img_path, os.path.join(dst_img_dir, img_file))
            # Empty label file
            with open(os.path.join(dst_lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                pass
            orphan_count += 1

        print(f"\n{dst_split}: {converted} annotated + {orphan_count} orphan = {converted + orphan_count} images")
        print(f"  Objects: {total_objects}")
        for cls in CLASSES:
            print(f"    {cls}: {class_counts[cls]}")

        stats[dst_split] = {
            "images": converted + orphan_count,
            "objects": total_objects,
            "class_counts": class_counts,
        }

    # Write data.yaml
    yaml_path = os.path.join(DST, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: train/images\n")
        f.write(f"val: valid/images\n")
        f.write(f"\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print(f"\ndata.yaml written to: {yaml_path}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
