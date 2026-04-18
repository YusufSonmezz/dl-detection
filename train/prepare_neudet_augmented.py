"""
NEU-DET Augmented Dataset Preparation Script

Orijinal veri setine dokunmadan yeni bir augmented veri seti olusturur.

Yapilan islemler:
1. Test goruntuleri train'e katilir (80:20 split)
2. Az temsil edilen siniflar augmentation ile dengelenir
3. Bos goruntulerin sayisi azaltilir (~50'ye)
4. Validation degistirilmez

Kullanim:
    python train/prepare_neudet_augmented.py [--dry-run]
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

try:
    import cv2
    import numpy as np
except ImportError:
    print("HATA: opencv-python ve numpy gerekli.")
    print("  pip install opencv-python numpy")
    sys.exit(1)

# ---------- Config ----------
SEED = 42
SRC_DIR = Path("train/dataset_neudet")
DST_DIR = Path("train/dataset_neudet_augmented")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TARGET_PER_CLASS = 270  # inclusion (en buyuk sinif) seviyesine dengeleme
MAX_EMPTY = 50          # Bos goruntu limiti

# Augmentation parametreleri
AUG_CONFIGS = [
    {"name": "hflip",    "hflip": True,  "rotation": 0,   "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,    "translate": (0, 0)},
    {"name": "rot10",    "hflip": False, "rotation": 10,  "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,    "translate": (0, 0)},
    {"name": "rot-10",   "hflip": False, "rotation": -10, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,    "translate": (0, 0)},
    {"name": "hsv",      "hflip": False, "rotation": 0,   "hsv_h": 10, "hsv_s": 30, "hsv_v": 30, "translate": (0, 0)},
    {"name": "trans",    "hflip": False, "rotation": 0,   "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,    "translate": (0.08, 0.08)},
    {"name": "combo1",   "hflip": True,  "rotation": 8,   "hsv_h": 5, "hsv_s": 15, "hsv_v": 15,  "translate": (0, 0)},
    {"name": "combo2",   "hflip": False, "rotation": -8,  "hsv_h": 8, "hsv_s": 20, "hsv_v": 20,  "translate": (0.05, 0.05)},
]


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Label dosyasini oku, [(class_id, x_center, y_center, w, h), ...] dondur."""
    bboxes = []
    if not label_path.exists():
        return bboxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bboxes.append((cls_id, x, y, w, h))
    return bboxes


def write_label_file(label_path: Path, bboxes: list[tuple[int, float, float, float, float]]):
    """Label dosyasina yaz."""
    with open(label_path, "w") as f:
        for cls_id, x, y, w, h in bboxes:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def augment_image(img: np.ndarray, bboxes: list, config: dict) -> tuple[np.ndarray, list]:
    """Goruntuve bbox'lari augment et. YOLO format: (cls, x_center, y_center, w, h) normalized."""
    h_img, w_img = img.shape[:2]
    new_bboxes = list(bboxes)  # copy
    result = img.copy()

    # 1. HSV jitter
    if config["hsv_h"] or config["hsv_s"] or config["hsv_v"]:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-config["hsv_h"], config["hsv_h"]), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-config["hsv_s"], config["hsv_s"]), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-config["hsv_v"], config["hsv_v"]), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 2. Rotation
    if config["rotation"] != 0:
        angle = config["rotation"]
        M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), angle, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        # Yeni boyutlar (rotate sonrasi crop yapmamak icin)
        # Ayni boyut, kenarlarda siyah kalabilir
        result = cv2.warpAffine(result, M, (w_img, h_img), borderValue=(114, 114, 114))

        # Bbox'lari rotate et
        rad = np.radians(-angle)  # opencv ile tutarli
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        rotated_bboxes = []
        for cls_id, xc, yc, bw, bh in new_bboxes:
            # Normalize -> pixel
            px, py = xc * w_img, yc * h_img
            # Merkeze gore rotate
            cx, cy = w_img / 2, h_img / 2
            dx, dy = px - cx, py - cy
            new_px = cos_r * dx - sin_r * dy + cx
            new_py = sin_r * dx + cos_r * dy + cy
            # Normalize'e geri
            new_xc = new_px / w_img
            new_yc = new_py / h_img
            # Kucuk acilarda w,h degismez (yaklasik)
            # Sinir kontrolu
            new_xc = max(0.001, min(0.999, new_xc))
            new_yc = max(0.001, min(0.999, new_yc))
            bw = min(bw, min(new_xc, 1 - new_xc) * 2)
            bh = min(bh, min(new_yc, 1 - new_yc) * 2)
            if bw > 0.01 and bh > 0.01:
                rotated_bboxes.append((cls_id, new_xc, new_yc, bw, bh))
        new_bboxes = rotated_bboxes

    # 3. Translate
    tx_frac, ty_frac = config["translate"]
    if tx_frac > 0 or ty_frac > 0:
        tx = random.uniform(-tx_frac, tx_frac)
        ty = random.uniform(-ty_frac, ty_frac)
        tx_px = int(tx * w_img)
        ty_px = int(ty * h_img)
        M_t = np.float32([[1, 0, tx_px], [0, 1, ty_px]])
        result = cv2.warpAffine(result, M_t, (w_img, h_img), borderValue=(114, 114, 114))

        translated_bboxes = []
        for cls_id, xc, yc, bw, bh in new_bboxes:
            new_xc = xc + tx
            new_yc = yc + ty
            # Sinirlari kontrol et
            x1 = max(0, new_xc - bw / 2)
            y1 = max(0, new_yc - bh / 2)
            x2 = min(1, new_xc + bw / 2)
            y2 = min(1, new_yc + bh / 2)
            new_bw = x2 - x1
            new_bh = y2 - y1
            if new_bw > 0.01 and new_bh > 0.01:
                translated_bboxes.append((cls_id, (x1 + x2) / 2, (y1 + y2) / 2, new_bw, new_bh))
        new_bboxes = translated_bboxes

    # 4. Horizontal flip (en son, bbox donusumu basit)
    if config["hflip"]:
        result = cv2.flip(result, 1)
        new_bboxes = [(cls_id, 1.0 - xc, yc, bw, bh)
                      for cls_id, xc, yc, bw, bh in new_bboxes]

    return result, new_bboxes


def get_class_image_map(label_dir: Path) -> dict[int, list[str]]:
    """Her sinif icin o sinifi iceren goruntu isimlerini dondur."""
    class_images: dict[int, list[str]] = {i: [] for i in range(len(CLASSES))}
    empty_images: list[str] = []

    for label_file in sorted(label_dir.glob("*.txt")):
        bboxes = parse_label_file(label_file)
        stem = label_file.stem
        if not bboxes:
            empty_images.append(stem)
            continue
        seen_classes = set()
        for cls_id, *_ in bboxes:
            if cls_id not in seen_classes:
                class_images[cls_id].append(stem)
                seen_classes.add(cls_id)

    return class_images, empty_images


def find_image_file(images_dir: Path, stem: str) -> Path | None:
    """Goruntu dosyasini bul (.jpg, .png, .bmp)."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


def copy_file_pair(stem: str, src_img_dir: Path, src_lbl_dir: Path,
                   dst_img_dir: Path, dst_lbl_dir: Path, new_stem: str = None):
    """Goruntu + label ciftini kopyala."""
    if new_stem is None:
        new_stem = stem

    img_src = find_image_file(src_img_dir, stem)
    lbl_src = src_lbl_dir / f"{stem}.txt"

    if img_src is None:
        return False

    img_dst = dst_img_dir / (new_stem + img_src.suffix)
    lbl_dst = dst_lbl_dir / f"{new_stem}.txt"

    shutil.copy2(img_src, img_dst)
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        # Bos label dosyasi olustur
        lbl_dst.touch()

    return True


def main():
    parser = argparse.ArgumentParser(description="NEU-DET Augmented Dataset Preparation")
    parser.add_argument("--dry-run", action="store_true", help="Dosya olusturmadan plani goster")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    if not SRC_DIR.exists():
        print(f"HATA: Kaynak dizin bulunamadi: {SRC_DIR}")
        sys.exit(1)

    # ---- Analiz ----
    print("=" * 60)
    print("NEU-DET Augmented Dataset Preparation")
    print("=" * 60)

    train_img_dir = SRC_DIR / "train" / "images"
    train_lbl_dir = SRC_DIR / "train" / "labels"
    test_img_dir = SRC_DIR / "test" / "images"
    test_lbl_dir = SRC_DIR / "test" / "labels"
    valid_img_dir = SRC_DIR / "valid" / "images"
    valid_lbl_dir = SRC_DIR / "valid" / "labels"

    # Train sinif dagilimi
    train_class_imgs, train_empty = get_class_image_map(train_lbl_dir)
    test_class_imgs, test_empty = get_class_image_map(test_lbl_dir)

    print("\n--- Mevcut Durum ---")
    print(f"Train: {sum(len(v) for v in train_class_imgs.values()) + len(train_empty)} goruntu ({len(train_empty)} bos)")
    print(f"Test:  {sum(len(v) for v in test_class_imgs.values()) + len(test_empty)} goruntu ({len(test_empty)} bos)")

    # Birlestirme sonrasi
    merged_class_imgs: dict[int, list[tuple[str, str]]] = {}  # cls -> [(stem, source_split), ...]
    for cls_id in range(len(CLASSES)):
        merged = [(s, "train") for s in train_class_imgs[cls_id]]
        merged += [(s, "test") for s in test_class_imgs[cls_id]]
        merged_class_imgs[cls_id] = merged

    merged_empty = [(s, "train") for s in train_empty] + [(s, "test") for s in test_empty]

    print("\n--- Birlestirme Sonrasi (Train + Test) ---")
    for cls_id, name in enumerate(CLASSES):
        count = len(merged_class_imgs[cls_id])
        need = max(0, TARGET_PER_CLASS - count)
        print(f"  {name:20s}: {count:4d} goruntu  (hedef: {TARGET_PER_CLASS}, augment: +{need})")
    print(f"  {'(bos)':20s}: {len(merged_empty):4d} goruntu  (hedef: {MAX_EMPTY}, cikarilacak: -{max(0, len(merged_empty) - MAX_EMPTY)})")

    # Augment ihtiyaci hesapla
    augment_plan: dict[int, int] = {}
    for cls_id in range(len(CLASSES)):
        current = len(merged_class_imgs[cls_id])
        augment_plan[cls_id] = max(0, TARGET_PER_CLASS - current)

    total_augment = sum(augment_plan.values())
    empty_to_keep = min(MAX_EMPTY, len(merged_empty))
    empty_to_remove = len(merged_empty) - empty_to_keep

    # Secilecek bos goruntuler
    random.shuffle(merged_empty)
    kept_empty = merged_empty[:empty_to_keep]

    # Toplam goruntu sayisi
    total_original = sum(len(v) for v in merged_class_imgs.values()) + empty_to_keep
    total_final = total_original + total_augment

    print(f"\n--- Plan ---")
    print(f"  Orijinal (train+test, bos azaltilmis): {total_original}")
    print(f"  Augment edilecek:                      +{total_augment}")
    print(f"  Final train seti:                      {total_final}")
    print(f"  Validation (degismez):                 360")

    if args.dry_run:
        print("\n[DRY RUN] Dosya olusturulmadi.")
        return

    # ---- Olustur ----
    print("\n--- Veri Seti Olusturuluyor ---")

    # Hedef dizinleri olustur
    dst_train_img = DST_DIR / "train" / "images"
    dst_train_lbl = DST_DIR / "train" / "labels"
    dst_valid_img = DST_DIR / "valid" / "images"
    dst_valid_lbl = DST_DIR / "valid" / "labels"

    for d in [dst_train_img, dst_train_lbl, dst_valid_img, dst_valid_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Tum sinifli goruntueri kopyala (train + test -> new train)
    copied_stems = set()
    copy_count = 0

    for cls_id in range(len(CLASSES)):
        for stem, source in merged_class_imgs[cls_id]:
            if stem in copied_stems:
                continue  # Ayni goruntu birden fazla sinif icerebilir
            copied_stems.add(stem)

            if source == "train":
                src_img = train_img_dir
                src_lbl = train_lbl_dir
            else:
                src_img = test_img_dir
                src_lbl = test_lbl_dir

            if copy_file_pair(stem, src_img, src_lbl, dst_train_img, dst_train_lbl):
                copy_count += 1

    # 2. Secilen bos goruntueri kopyala
    for stem, source in kept_empty:
        if stem in copied_stems:
            continue
        copied_stems.add(stem)

        if source == "train":
            src_img = train_img_dir
            src_lbl = train_lbl_dir
        else:
            src_img = test_img_dir
            src_lbl = test_lbl_dir

        if copy_file_pair(stem, src_img, src_lbl, dst_train_img, dst_train_lbl):
            copy_count += 1

    print(f"  Orijinal goruntuler kopyalandi: {copy_count}")

    # 3. Augmentation
    aug_count = 0
    for cls_id in range(len(CLASSES)):
        needed = augment_plan[cls_id]
        if needed == 0:
            continue

        source_stems = merged_class_imgs[cls_id]
        print(f"  {CLASSES[cls_id]:20s}: {needed} augment uretiliyor...", end=" ")

        generated = 0
        aug_idx = 0
        attempts = 0
        max_attempts = needed * 10  # sonsuz dongu koruması

        while generated < needed and attempts < max_attempts:
            attempts += 1
            # Kaynak goruntu sec (round-robin)
            stem, source = source_stems[aug_idx % len(source_stems)]
            aug_idx += 1

            # Augmentation config sec
            config = AUG_CONFIGS[generated % len(AUG_CONFIGS)]

            # Kaynak dosyalari bul
            if source == "train":
                src_img_dir_cur = train_img_dir
                src_lbl_dir_cur = train_lbl_dir
            else:
                src_img_dir_cur = test_img_dir
                src_lbl_dir_cur = test_lbl_dir

            img_path = find_image_file(src_img_dir_cur, stem)
            if img_path is None:
                continue

            lbl_path = src_lbl_dir_cur / f"{stem}.txt"
            bboxes = parse_label_file(lbl_path)
            if not bboxes:
                continue

            # Goruntuleyi oku ve augment et
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            aug_img, aug_bboxes = augment_image(img, bboxes, config)

            if not aug_bboxes:
                continue

            # Kaydet
            new_stem = f"{stem}_aug_{config['name']}_{generated}"
            img_ext = img_path.suffix
            cv2.imwrite(str(dst_train_img / f"{new_stem}{img_ext}"), aug_img)
            write_label_file(dst_train_lbl / f"{new_stem}.txt", aug_bboxes)

            generated += 1

        aug_count += generated
        print(f"{generated} uretildi")

    print(f"  Toplam augment: {aug_count}")

    # 4. Validation'i kopyala (degistirmeden)
    valid_count = 0
    for label_file in sorted(valid_lbl_dir.glob("*.txt")):
        stem = label_file.stem
        if copy_file_pair(stem, valid_img_dir, valid_lbl_dir, dst_valid_img, dst_valid_lbl):
            valid_count += 1

    print(f"  Validation kopyalandi: {valid_count}")

    # ---- Dogrulama ----
    print("\n--- Dogrulama ---")
    for split_name, lbl_dir in [("train", dst_train_lbl), ("valid", dst_valid_lbl)]:
        class_counts = Counter()
        total_imgs = 0
        empty_count = 0
        for lf in lbl_dir.glob("*.txt"):
            total_imgs += 1
            bboxes = parse_label_file(lf)
            if not bboxes:
                empty_count += 1
                continue
            seen = set()
            for cls_id, *_ in bboxes:
                if cls_id not in seen:
                    class_counts[cls_id] += 1
                    seen.add(cls_id)

        print(f"\n  {split_name.upper()}: {total_imgs} goruntu ({empty_count} bos)")
        for cls_id, name in enumerate(CLASSES):
            cnt = class_counts.get(cls_id, 0)
            print(f"    {name:20s}: {cnt:4d}")

    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print(f"Yeni veri seti: {DST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
