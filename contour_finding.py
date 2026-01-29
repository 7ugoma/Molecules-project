import cv2
import numpy as np

my_photo = cv2.imread('archive/cut3.png')
img_grey = cv2.cvtColor(my_photo, cv2.COLOR_BGR2GRAY)
filterd_image = cv2.GaussianBlur(img_grey, (5, 5), 0)

binary_adaptive = cv2.adaptiveThreshold(filterd_image, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

contours, hierarchy = cv2.findContours(binary_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_area = 30
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


def is_circular_by_vertices(contour, epsilon_factor=0.03, min_vertices=6, max_vertices=20):
    if len(contour) < 5:
        return False
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return min_vertices <= len(approx) <= max_vertices


circular_contours = []
for cnt in filtered_contours:
    if is_circular_by_vertices(cnt):
        circular_contours.append(cnt)

circular_contours_sorted = sorted(circular_contours, key=cv2.contourArea, reverse=True)

if len(circular_contours_sorted) > 0:
    reference_cnt = circular_contours_sorted[0]
    reference_area = cv2.contourArea(reference_cnt)
    print(f"Площадь эталонного круглого контура: {reference_area}")
else:
    print("Не найдено круглых контуров")
    exit()

tolerance_percent = 97
tolerance_factor = tolerance_percent / 100.0

similar_circular_contours = []
for cnt in circular_contours:
    area = cv2.contourArea(cnt)
    if reference_area * (1 - tolerance_factor) <= area <= reference_area * (1 + tolerance_factor):
        similar_circular_contours.append(cnt)

print(f"Найдено {len(similar_circular_contours)} круглых контуров схожего размера")

img_result = my_photo.copy()
img_circular_only = np.zeros_like(img_grey)
img_all_circular = np.zeros_like(img_grey)

cv2.drawContours(img_all_circular, circular_contours, -1, (255, 255, 255), 1)
cv2.drawContours(img_circular_only, similar_circular_contours, -1, (255, 255, 255), 1)

for i, cnt in enumerate(similar_circular_contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius_int = int(radius)

    cv2.circle(img_result, center, radius_int, (0, 255, 0), 2)
    cv2.circle(img_result, center, 3, (0, 0, 255), -1)

    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        print(f"Круглый контур {i + 1}: центр=({x:.1f}, {y:.1f}), радиус={radius:.1f}, круглость={circularity:.3f}")

cv2.imshow('Original', my_photo)
cv2.imshow('All circular contours', img_all_circular)
cv2.imshow('Similar circular contours', img_circular_only)
cv2.imshow('Result with circles', img_result)

cv2.waitKey()
cv2.destroyAllWindows()