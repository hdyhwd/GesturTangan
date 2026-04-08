import cv2
import numpy as np

# =========================
# 1. INPUT GAMBAR
# =========================
img = cv2.imread('dataset/OpenHand/g3.jpeg')

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

img = cv2.resize(img, (300, 300))

# =========================
# 2. GRAYSCALE
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 3. BLUR (LEBIH KUAT)
# =========================
blur = cv2.GaussianBlur(gray, (7,7), 0)

# =========================
# 4. THRESHOLD (OTSU + INV OPSIONAL)
# =========================
_, thresh = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 👉 kalau hasil kebalik, ganti jadi:
# cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

# =========================
# 5. MORPHOLOGY (LEBIH KUAT)
# =========================
kernel = np.ones((7,7), np.uint8)

morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# =========================
# 6. CONTOUR
# =========================
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contour = img.copy()

if contours:
    # filter contour kecil (biar noise hilang)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]

    if contours:
        c = max(contours, key=cv2.contourArea)

        # =========================
        # 7. FEATURE EXTRACTION
        # =========================
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        x, y, w, h = cv2.boundingRect(c)
        ratio = h / w if w != 0 else 0

        # =========================
        # VISUALISASI
        # =========================
        cv2.drawContours(img_contour, [c], -1, (0,255,0), 2)
        cv2.rectangle(img_contour, (x,y), (x+w, y+h), (255,0,0), 2)

        # =========================
        # TEXT FEATURE DI GAMBAR
        # =========================
        cv2.putText(img_contour, f"Area: {int(area)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.putText(img_contour, f"Perim: {int(perimeter)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.putText(img_contour, f"Ratio: {round(ratio,2)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # =========================
        # PRINT KE TERMINAL
        # =========================
        print("=== FEATURE EXTRACTION ===")
        print("Area      :", area)
        print("Perimeter :", perimeter)
        print("Ratio     :", ratio)

    else:
        print("Semua contour terlalu kecil (noise)")
else:
    print("Contour tidak ditemukan!")

# =========================
# 8. TAMPILKAN SEMUA
# =========================
cv2.imshow("1. Original", img)
cv2.imshow("2. Grayscale", gray)
cv2.imshow("3. Blur", blur)
cv2.imshow("4. Threshold", thresh)
cv2.imshow("5. Morphology", morph)
cv2.imshow("6. Contour + Feature", img_contour)

cv2.waitKey(0)
cv2.destroyAllWindows()