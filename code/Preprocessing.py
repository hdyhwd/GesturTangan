import cv2
import numpy as np
import pandas as pd
import os

# ======================================
# KONFIGURASI
# ======================================
DATASET_PATH = "../dataset"
IMAGE_SIZE = (300, 300)

# ======================================
# MENAMPUNG DATA
# ======================================
data = []

jumlah_berhasil = 0
jumlah_gagal = 0

# ======================================
# LOOP SEMUA FOLDER
# ======================================
for label in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nMemproses kelas: {label}")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        file_path = os.path.join(
            folder_path,
            filename
        )

        # ======================================
        # BACA GAMBAR
        # ======================================
        img = cv2.imread(file_path)

        if img is None:
            jumlah_gagal += 1
            print("Gagal :", filename)
            continue

        # ======================================
        # SAMAKAN UKURAN
        # ======================================
        img = cv2.resize(
            img,
            IMAGE_SIZE
        )

        # ======================================
        # PREPROCESSING
        # ======================================
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (7, 7),
            0
        )

        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )

        kernel = np.ones(
            (7, 7),
            np.uint8
        )

        morph = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel
        )

        morph = cv2.morphologyEx(
            morph,
            cv2.MORPH_OPEN,
            kernel
        )

        # ======================================
        # CONTOUR
        # ======================================
        contours, _ = cv2.findContours(
            morph,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = [
            c for c in contours
            if cv2.contourArea(c) > 1000
        ]

        if len(contours) == 0:
            jumlah_gagal += 1
            continue

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        # ======================================
        # FITUR
        # ======================================
        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        x, y, w, h = cv2.boundingRect(
            cnt
        )

        ratio = (
            h / w
            if w != 0
            else 0
        )

        hull_points = cv2.convexHull(
            cnt
        )

        hull_area = cv2.contourArea(
            hull_points
        )

        solidity = (
            area / hull_area
            if hull_area != 0
            else 0
        )

        rect_area = w * h

        extent = (
            area / rect_area
            if rect_area != 0
            else 0
        )

        convexity = (
            hull_area / area
            if area != 0
            else 0
        )

        # ======================================
        # HITUNG JARI
        # ======================================
        finger_count = 0

        hull = cv2.convexHull(
            cnt,
            returnPoints=False
        )

        if len(hull) > 3:

            defects = cv2.convexityDefects(
                cnt,
                hull
            )

            if defects is not None:

                count_defects = 0

                for i in range(
                    defects.shape[0]
                ):

                    s, e, f, d = defects[i, 0]

                    start = tuple(
                        cnt[s][0]
                    )

                    end = tuple(
                        cnt[e][0]
                    )

                    far = tuple(
                        cnt[f][0]
                    )

                    a = np.linalg.norm(
                        np.array(end) -
                        np.array(start)
                    )

                    b = np.linalg.norm(
                        np.array(far) -
                        np.array(start)
                    )

                    c = np.linalg.norm(
                        np.array(end) -
                        np.array(far)
                    )

                    if b * c == 0:
                        continue

                    angle = np.arccos(
                        (b**2 + c**2 - a**2)
                        /
                        (2 * b * c)
                    )

                    if angle <= np.pi / 2:
                        count_defects += 1

                finger_count = (
                    count_defects + 1
                )

        # ======================================
        # SIMPAN DATA
        # ======================================
        data.append([
            filename,
            area,
            perimeter,
            ratio,
            hull_area,
            solidity,
            extent,
            convexity,
            finger_count,
            label
        ])

        jumlah_berhasil += 1

# ======================================
# DATAFRAME
# ======================================
df = pd.DataFrame(
    data,
    columns=[
        "filename",
        "area",
        "perimeter",
        "ratio",
        "hull_area",
        "solidity",
        "extent",
        "convexity",
        "finger_count",
        "label"
    ]
)

# ======================================
# SIMPAN CSV
# ======================================
df.to_csv(
    "dataset_fitur.csv",
    index=False
)

# ======================================
# HASIL
# ======================================
print("\n===================================")
print("Jumlah berhasil :", jumlah_berhasil)
print("Jumlah gagal    :", jumlah_gagal)
print("Total data      :", len(df))
print("===================================")

print("\nJumlah per kelas:")
print(df["label"].value_counts())

print("\nContoh data:")
print(df.head(20))

print("\nCSV berhasil disimpan:")
print("dataset_fitur.csv")