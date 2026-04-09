from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Функция вычисления "заполненности" круглей
# принимает путь до картинки, радиус, коорды центра круга
def calculate_area(way, r, centy, centx):
    img = cv2.imread(way)  # читаем изображение
    imgbgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # перевод в RGB (для matplotlib)

    # выделяем только белый, по нему будем фильтровать
    low_color = (235, 235, 235)
    high_color = (255, 255, 255)

    # Центр и радиус круга
    cx, cy = centx, centy

    # Создаем маску круга (черное изображение с белым кругом)
    mask_circle = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask_circle, (cx, cy), r, 255, -1)

    # Оставляем только область внутри круга
    masked_image = cv2.bitwise_and(img, img, mask=mask_circle)

    # Создаем маску белых пикселей внутри круга
    color_mask = cv2.inRange(masked_image, low_color, high_color)

    # Показываем маску (для дебага)
    plt.imshow(color_mask)
    # plt.show()

    # Считаем площадь круга (в пикселях)
    circle_area = cv2.countNonZero(mask_circle)

    # Считаем количество "белых" пикселей
    color_area = cv2.countNonZero(color_mask)

    # Возвращаем долю белого внутри круга
    return color_area / circle_area

# ф-я вычисления площадей, принимает путь до картинки и название используемой модели yolo
# возвращает словарь: ключ - номер по порядку, внутри список с процентом заполненности, кортежом (x, y) центра, радиус
# и возвращает картинку с пронумерованными круглями
def areas(img_way, model_name='best.pt'):
    square_list = {}
    # Загружаем обученную YOLO модель
    model = YOLO("best.pt")

    # Путь к изображению
    img_way = img_way

    # Прогоняем изображение через модель (получаем детекции)
    results = model(img_way)

    quant = 0  # количество найденных (уникальных) круглей

    # Картинка, на которой будем рисовать номера
    img_res = cv2.imread(img_way)

    # Множество координат центров (для фильтрации дубликатов)
    koords = set()

    # Флаги:
    # fl1 — нашли дубликат (слишком близко к уже найденному)
    # fl2 — круг выходит за границы изображения
    fl1, fl2 = False, False

    # Перебираем результаты модели
    for result in results:

        # Проверяем, есть ли bounding boxes
        if result.boxes is not None:

            # Получаем координаты прямоугольников [x1, y1, x2, y2]
            boxes = result.boxes.xyxy.cpu().numpy()

            # Перебираем каждый найденный объект
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box

                # Вычисляем центр прямоугольника
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Радиус круга (примерно половина среднего размера bbox)
                radius = int((x2 - x1 + y2 - y1) / 4)

                # Проверка на дубликаты (слишком близкие центры)
                for xpovt, ypovt in koords:
                    # np.hypot — расстояние между точками
                    if np.hypot(cx - xpovt, cy - ypovt) < 10:
                        fl1 = True

                # Добавляем текущий центр в множество
                koords.add((int(cx), int(cy)))

                # Загружаем изображение (для проверки границ)
                img = cv2.imread(img_way)
                imgh, imgw = img.shape[:2]

                # Проверка: не выходит ли круг за границы изображения
                if cx + radius > imgw or cy + radius > imgh or cx - radius < 0 or cy - radius < 0:
                    fl2 = True

                # Если это дубликат или выходит за границы — пропускаем
                if fl1 == True or fl2 == True:
                    # print("ababa")  # можно заменить на нормальный лог, но не надо
                    fl1, fl2 = False, False
                    continue

                # Если всё ок — считаем
                print(f"Объект {quant + 1}:")

                # Считаем процент заполнения белым
                ploshad = calculate_area(img_way, int(radius), int(cy), int(cx))

                # Подписываем номер круга на изображении
                cv2.putText(
                    img_res,
                    str(quant + 1),
                    (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.5,
                    (0, 0, 0),
                    2
                )

                # Выводим процент заполненности

                print(f"{(round(100 - ploshad * 100, 2))}% заполнено")
                square_list[quant + 1] = [round(100 - ploshad * 100, 2), (cx, cy), radius]
                # Отладочная инфа
                # print(f"  Центр: ({cx}, {cy})", imgh, imgw)
                # print(f"  Радиус: {radius}", cx + radius, cy + radius)

                # Визуализация детекций YOLO
                plotted = result.plot()

                quant += 1

        # Показываем окна с результатами
    #  cv2.imshow("masks", plotted)
    #  cv2.imshow("Detection", img_res)
    #  cv2.waitKey(0)

    return square_list, img_res


def main():
    img_way = ''            #путь до картинки
    model_name = ''                  #название модельки
    sqr_list, img_res = areas(img_way, model_name)
    print(sqr_list)
    cv2.imshow('result', img_res)
    cv2.waitKey(0)


# Закрываем окна
main()
cv2.destroyAllWindows()
