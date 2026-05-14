import sys
import os
import cv2
import numpy as np
import torch
import gc
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QFileDialog, QMessageBox, QSlider, QSplitter,
                             QFrame, QProgressBar, QStatusBar, QToolBar,
                             QAction, QGroupBox, QTableWidget, QTableWidgetItem,
                             QScrollArea, QHeaderView)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QBrush, QWheelEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QPoint, QEvent
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ========== ФУНКЦИИ ИЗ molecs.py ==========

def calculate_fill_percentage(way, r, centy, centx):
    """Функция вычисления "заполненности" сфероида"""
    img = cv2.imread(way)

    low_color = (235, 235, 235)
    high_color = (255, 255, 255)

    cx, cy = centx, centy

    mask_circle = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask_circle, (cx, cy), r, 255, -1)

    masked_image = cv2.bitwise_and(img, img, mask=mask_circle)
    color_mask = cv2.inRange(masked_image, low_color, high_color)

    circle_area = cv2.countNonZero(mask_circle)
    color_area = cv2.countNonZero(color_mask)

    return color_area / circle_area if circle_area > 0 else 0


def process_image_with_yolo(img_way, model_name='best.pt'):
    """
    Функция обработки изображения с помощью YOLO
    Возвращает словарь с информацией о сфероидах и изображение с нумерацией
    """
    square_list = {}

    model = YOLO(model_name)
    results = model(img_way)

    quant = 0
    img_res = cv2.imread(img_way)
    koords = set()

    fl1, fl2 = False, False

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                radius = int((x2 - x1 + y2 - y1) / 4)

                # Проверка на дубликаты
                for xpovt, ypovt in koords:
                    if np.hypot(cx - xpovt, cy - ypovt) < 10:
                        fl1 = True

                koords.add((int(cx), int(cy)))

                img = cv2.imread(img_way)
                imgh, imgw = img.shape[:2]

                # Проверка на выход за границы
                if cx + radius > imgw or cy + radius > imgh or cx - radius < 0 or cy - radius < 0:
                    fl2 = True

                if fl1 or fl2:
                    fl1, fl2 = False, False
                    continue

                # Вычисляем заполненность
                ploshad = calculate_fill_percentage(img_way, int(radius), int(cy), int(cx))
                fill_percent = round(100 - ploshad * 100, 2)

                # Нумерация в стиле 4-го изображения: синий кружок с номером
                cv2.circle(img_res, (int(cx), int(cy)), radius + 5, (255, 0, 0), 3)  # Синяя окружность
                cv2.circle(img_res, (int(cx), int(cy)), 20, (255, 0, 0), -1)  # Синий кружок
                cv2.putText(
                    img_res,
                    str(quant + 1),
                    (int(cx) - 10, int(cy) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                square_list[quant + 1] = {
                    'fill_percentage': fill_percent,
                    'center': (cx, cy),
                    'radius': radius,
                    'bbox': (x1, y1, x2, y2)
                }

                quant += 1

    return square_list, img_res, results


def draw_russian_text(image, text, position, font_size=20, color=(255, 255, 255),
                      outline_color=(0, 0, 0), outline_width=2):
    """
    Рисует русский текст на изображении с обводкой для лучшей читаемости
    """
    # Конвертируем OpenCV BGR в RGB для PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Загружаем шрифт с поддержкой кириллицы
    font = None
    font_paths = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf"
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue

    if font is None:
        font = ImageFont.load_default()

    # Рисуем обводку
    if outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((position[0] + dx, position[1] + dy), text,
                              font=font, fill=tuple(outline_color))

    # Рисуем основной текст
    draw.text(position, text, font=font, fill=tuple(color))

    # Конвертируем обратно в OpenCV BGR
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ========== КЛАССЫ ИНТЕРФЕЙСА ==========

class ProcessingThread(QThread):
    """Поток для обработки изображения"""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, image_path, confidence_threshold=0.5):
        super().__init__()
        self.image_path = image_path
        self.confidence_threshold = confidence_threshold

    def run(self):
        self.status.emit("Запуск модели YOLO...")
        self.progress.emit(20)

        try:
            # Используем функцию из molecs.py
            square_list, numbered_image, yolo_results = process_image_with_yolo(self.image_path)

            self.progress.emit(50)
            self.status.emit("Обработка результатов...")

            # Загружаем оригинальное изображение
            original_image = cv2.imread(self.image_path)

            # Получаем маски из YOLO
            masks_data = []

            if yolo_results and yolo_results[0].masks is not None:
                masks = yolo_results[0].masks.data.cpu().numpy()
                boxes = yolo_results[0].boxes
                classes = yolo_results[0].boxes.cls.cpu().numpy() if hasattr(yolo_results[0].boxes,
                                                                             'cls') else np.zeros(len(boxes))

                # Сопоставляем маски с объектами из square_list
                h, w = original_image.shape[:2]

                for obj_id, obj_data in square_list.items():
                    center = obj_data['center']

                    # Находим ближайшую маску
                    best_mask_idx = None
                    min_dist = float('inf')

                    for i, mask in enumerate(masks):
                        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        moments = cv2.moments(mask_binary)
                        if moments["m00"] != 0:
                            mask_cx = int(moments["m10"] / moments["m00"])
                            mask_cy = int(moments["m01"] / moments["m00"])
                            dist = np.hypot(center[0] - mask_cx, center[1] - mask_cy)
                            if dist < min_dist and dist < obj_data['radius']:
                                min_dist = dist
                                best_mask_idx = i

                    if best_mask_idx is not None:
                        mask = masks[best_mask_idx]
                        conf = float(boxes[best_mask_idx].conf) if best_mask_idx < len(boxes) else 0.5

                        if conf >= self.confidence_threshold:
                            class_idx = int(classes[best_mask_idx]) if best_mask_idx < len(classes) else 0
                            class_name = yolo_results[0].names[class_idx] if hasattr(yolo_results[0],
                                                                                     'names') else f"Class {class_idx}"

                            masks_data.append({
                                'mask': mask,
                                'conf': conf,
                                'index': obj_id,
                                'class_name': class_name,
                                'class_idx': class_idx,
                                'bbox': obj_data['bbox'],
                                'fill_percentage': obj_data['fill_percentage'],
                                'center': center,
                                'radius': obj_data['radius']
                            })

            self.progress.emit(80)
            self.status.emit("Визуализация...")

            # Создаем изображение с масками YOLO
            yolo_plotted = yolo_results[0].plot() if yolo_results else original_image

            self.progress.emit(100)
            self.status.emit(f"Готово! Найдено {len(masks_data)} сфероидов")

            self.finished.emit({
                'masks': masks_data,
                'yolo_image': yolo_plotted,
                'numbered_image': numbered_image,
                'original_image': original_image,
                'square_list': square_list
            })

        except Exception as e:
            self.status.emit(f"Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()
            self.finished.emit(None)


class ZoomableImageLabel(QLabel):
    """Виджет для отображения изображения с возможностью зума и панорамирования"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: none; background: #2b2b2b;")
        self.setMinimumSize(400, 400)
        self.setScaledContents(False)

        self.original_pixmap = None
        self.current_pixmap = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        self.is_dragging = False
        self.last_mouse_pos = None

        self.setMouseTracking(True)

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()

    def update_display(self):
        if self.original_pixmap is None or self.original_pixmap.isNull():
            self.clear()
            self.setText("Нет изображения")
            return

        try:
            if self.zoom <= 1.0:
                scaled = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled)
                self.current_pixmap = scaled
            else:
                orig_w = self.original_pixmap.width()
                orig_h = self.original_pixmap.height()

                new_w = int(orig_w * self.zoom)
                new_h = int(orig_h * self.zoom)

                max_size = 3000
                if new_w > max_size or new_h > max_size:
                    scale = min(max_size / new_w, max_size / new_h)
                    new_w = int(new_w * scale)
                    new_h = int(new_h * scale)
                    self.zoom = new_w / orig_w

                scaled = self.original_pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                widget_w = self.width()
                widget_h = self.height()

                x = self.pan_x
                y = self.pan_y

                max_pan_x = max(0, scaled.width() - widget_w)
                max_pan_y = max(0, scaled.height() - widget_h)
                x = max(0, min(x, max_pan_x))
                y = max(0, min(y, max_pan_y))

                self.pan_x = x
                self.pan_y = y

                if scaled.width() > widget_w or scaled.height() > widget_h:
                    cropped = scaled.copy(x, y, min(widget_w, scaled.width() - x), min(widget_h, scaled.height() - y))
                    self.setPixmap(cropped)
                    self.current_pixmap = cropped
                else:
                    self.setPixmap(scaled)
                    self.current_pixmap = scaled
        except Exception as e:
            print(f"Error in display update: {e}")
            self.setPixmap(self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self.original_pixmap is None:
            return

        old_zoom = self.zoom

        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom *= 0.9

        self.zoom = max(0.5, min(self.zoom, 3.0))

        if self.zoom != old_zoom and self.zoom > 1.0:
            zoom_ratio = self.zoom / old_zoom
            self.pan_x = int(self.pan_x * zoom_ratio)
            self.pan_y = int(self.pan_y * zoom_ratio)

        self.update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.zoom > 1.0:
            self.drag_start = event.pos()
            self.last_mouse_pos = event.pos()
            self.is_dragging = True
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging and self.drag_start and self.zoom > 1.0:
            if self.last_mouse_pos:
                delta = event.pos() - self.last_mouse_pos
                self.pan_x -= delta.x()
                self.pan_y -= delta.y()

                if self.original_pixmap:
                    new_w = int(self.original_pixmap.width() * self.zoom)
                    new_h = int(self.original_pixmap.height() * self.zoom)
                    widget_w = self.width()
                    widget_h = self.height()

                    max_pan_x = max(0, new_w - widget_w)
                    max_pan_y = max(0, new_h - widget_h)

                    self.pan_x = max(0, min(self.pan_x, max_pan_x))
                    self.pan_y = max(0, min(self.pan_y, max_pan_y))

                self.last_mouse_pos = event.pos()
                self.update_display()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.drag_start = None
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def clear(self):
        self.original_pixmap = None
        self.current_pixmap = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        super().clear()
        self.setText("Нет изображения")

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()


class ImageContainer(QWidget):
    """Контейнер для изображения с прокруткой и поддержкой зума"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = ZoomableImageLabel()
        layout.addWidget(self.image_label)

    def set_image(self, pixmap):
        self.image_label.set_image(pixmap)

    def clear(self):
        self.image_label.clear()

    def reset_view(self):
        self.image_label.reset_view()

    def get_image_label(self):
        return self.image_label


class DropZone(QLabel):
    """Область для Drag & Drop файлов"""

    file_dropped = pyqtSignal(str)

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #bbb;
                background: #f9f9f9;
                border-radius: 10px;
                font-size: 14px;
                color: #666;
                padding: 20px;
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #0277bd;
                    background: #e1f5fe;
                    border-radius: 10px;
                    font-size: 14px;
                    color: #0277bd;
                    padding: 20px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #bbb;
                background: #f9f9f9;
                border-radius: 10px;
                font-size: 14px;
                color: #666;
                padding: 20px;
            }
        """)

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.file_dropped.emit(file_path)
            else:
                QMessageBox.warning(self, "Ошибка", "Пожалуйста, используйте изображения (jpg, png, tif).")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #bbb;
                background: #f9f9f9;
                border-radius: 10px;
                font-size: 14px;
                color: #666;
                padding: 20px;
            }
        """)


class ResultsTableWidget(QWidget):
    """Виджет с таблицей результатов"""

    item_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header_label = QLabel("📊 Результаты анализа сфероидов")
        self.header_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2196f3;
                color: white;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.header_label)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Класс", "Уверенность", "Насыщенность %", "Площадь", "Радиус",
             "Центр X", "Центр Y"
        ])

        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(False)



        header = self.table.horizontalHeader()
        header.setSectionsMovable(False)
        header.setSectionResizeMode(QHeaderView.Fixed)

        column_widths = [110, 100, 120, 90, 80, 90, 90]
        for i, width in enumerate(column_widths):
            self.table.setColumnWidth(i, width)

        font = QFont()
        font.setPointSize(10)
        self.table.setFont(font)

        self.table.itemSelectionChanged.connect(self.on_selection_changed)

        layout.addWidget(self.table)

        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
            }
        """)
        stats_layout = QHBoxLayout(self.stats_frame)

        self.total_count_label = QLabel("Всего объектов: 0")
        self.total_count_label.setStyleSheet("font-weight: bold; padding: 5px;")
        stats_layout.addWidget(self.total_count_label)

        stats_layout.addStretch()

        self.avg_fill_label = QLabel("Ср. насыщенность: 0%")
        self.avg_fill_label.setStyleSheet("padding: 5px;")
        stats_layout.addWidget(self.avg_fill_label)

        self.avg_confidence_label = QLabel("Ср. уверенность: 0")
        self.avg_confidence_label.setStyleSheet("padding: 5px;")
        stats_layout.addWidget(self.avg_confidence_label)

        layout.addWidget(self.stats_frame)

        reset_zoom_btn = QPushButton("🔄 Сбросить зум во всех окнах")
        reset_zoom_btn.clicked.connect(self.reset_all_zooms)
        layout.addWidget(reset_zoom_btn)

    def reset_all_zooms(self):
        main_window = self.window()
        if hasattr(main_window, 'reset_all_zooms'):
            main_window.reset_all_zooms()

    def update_data(self, masks_data, original_image=None):
        self.table.setRowCount(0)

        if not masks_data:
            self.update_statistics(0, [], [])
            return

        h, w = original_image.shape[:2] if original_image is not None else (1, 1)

        confidences = []
        fill_percentages = []

        for i, mask_data in enumerate(masks_data):
            row = self.table.rowCount()
            self.table.insertRow(row)

            fill_pct = mask_data.get('fill_percentage', 0.0)
            center = mask_data.get('center', (0, 0))
            radius = mask_data.get('radius', 0)

            area = self.calculate_mask_area(mask_data['mask'], w, h)
            perimeter = self.calculate_mask_perimeter(mask_data['mask'], w, h)

            confidences.append(mask_data['conf'])
            fill_percentages.append(fill_pct)

            self.table.setItem(row, 0, QTableWidgetItem(mask_data['class_name']))

            conf_item = QTableWidgetItem(f"{mask_data['conf']:.3f}")
            if mask_data['conf'] > 0.7:
                conf_item.setForeground(QBrush(QColor(76, 175, 80)))
            elif mask_data['conf'] > 0.4:
                conf_item.setForeground(QBrush(QColor(255, 152, 0)))
            else:
                conf_item.setForeground(QBrush(QColor(244, 67, 54)))
            self.table.setItem(row, 1, conf_item)

            fill_item = QTableWidgetItem(f"{fill_pct:.1f}%")
            if fill_pct > 70:
                fill_item.setForeground(QBrush(QColor(244, 67, 54)))
            elif fill_pct > 40:
                fill_item.setForeground(QBrush(QColor(255, 152, 0)))
            else:
                fill_item.setForeground(QBrush(QColor(76, 175, 80)))
            self.table.setItem(row, 2, fill_item)

            self.table.setItem(row, 3, QTableWidgetItem(f"{area:.0f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{radius:.1f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{center[0]:.1f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{center[1]:.1f}"))

            self.table.item(row, 0).setData(Qt.UserRole, i)

        self.update_statistics(len(masks_data), confidences, fill_percentages)
        self.table.resizeRowsToContents()

    def calculate_mask_area(self, mask, img_w, img_h):
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask_resized = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        return float(np.sum(mask_resized > 0.5))

    def calculate_mask_perimeter(self, mask, img_w, img_h):
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask_resized = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
        return 0.0

    def update_statistics(self, total_count, confidences, fill_percentages):
        self.total_count_label.setText(f"📈 Всего объектов: {total_count}")
        if confidences:
            self.avg_confidence_label.setText(f"🎯 Ср. уверенность: {np.mean(confidences):.3f}")
        else:
            self.avg_confidence_label.setText("🎯 Ср. уверенность: 0")
        if fill_percentages:
            self.avg_fill_label.setText(f"💧 Ср. насыщенность: {np.mean(fill_percentages):.1f}%")
        else:
            self.avg_fill_label.setText("💧 Ср. насыщенность: 0%")

    def on_selection_changed(self):
        selected_rows = self.table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            item = self.table.item(row, 0)
            if item:
                self.item_selected.emit(item.data(Qt.UserRole))

    def clear(self):
        self.table.setRowCount(0)
        self.update_statistics(0, [], [])


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spheroid Segmentation with YOLO")
        self.setGeometry(100, 100, 1500, 900)

        self.current_image_path = None
        self.current_masks = []
        self.original_image = None
        self.yolo_masks_image = None
        self.numbered_masks_image = None
        self.square_list = {}

        self.setup_ui()
        self.create_menu_bar()
        self.setup_status_bar()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        info_frame = QFrame()
        info_frame.setMaximumHeight(45)
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin: 2px;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(6, 2, 6, 2)

        info_icon = QLabel("🖱️")
        info_icon.setStyleSheet("font-size: 14px;")
        info_layout.addWidget(info_icon)

        info_text = QLabel(
            "Управление: Колесико мыши - зум (0.5x - 3x) | Левая кнопка + перетаскивание - панорамирование | Ctrl+R - сброс зума"
        )
        info_text.setStyleSheet("font-size: 12px; color: #333")
        info_layout.addWidget(info_text, 1)

        main_layout.addWidget(info_frame)

        main_splitter = QSplitter(Qt.Horizontal)

        # Левая панель - изображения
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        images_container = QWidget()
        self.image_grid = QGridLayout(images_container)
        self.image_grid.setSpacing(10)

        self.image_containers = []
        titles = ["Оригинал", "Маски YOLO", "Пронумерованные сфероиды", "Выбранный сфероид"]

        for i in range(4):
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 5px; background: #2b2b2b; }")
            frame.setMinimumSize(400, 400)

            layout = QVBoxLayout(frame)
            layout.setContentsMargins(5, 5, 5, 5)

            title_label = QLabel(titles[i])
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px; background: #f0f0f0; padding: 5px;")
            title_label.setFixedHeight(30)
            layout.addWidget(title_label)

            img_container = ImageContainer()
            layout.addWidget(img_container)

            row, col = i // 2, i % 2
            self.image_grid.addWidget(frame, row, col)
            self.image_containers.append(img_container)

        self.image_containers[3].set_image(QPixmap())
        self.image_containers[3].image_label.setText("Выберите сфероид из таблицы")

        left_layout.addWidget(images_container)

        # Правая панель - управление
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.drop_zone = DropZone("📁 Перетащите изображение сюда\nили используйте кнопку ниже")
        self.drop_zone.file_dropped.connect(self.load_image)
        right_layout.addWidget(self.drop_zone)

        self.upload_btn = QPushButton("📥 Выбрать файл через проводник")
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                background-color: #e1f5fe;
                border: 1px solid #0277bd;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #b3e5fc; }
        """)
        self.upload_btn.clicked.connect(self.open_file_dialog)
        right_layout.addWidget(self.upload_btn)

        control_group = QGroupBox("Управление моделью")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        control_layout = QVBoxLayout(control_group)

        self.run_btn = QPushButton("▶ Запустить анализ")
        self.run_btn.setFixedHeight(50)
        self.run_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        control_layout.addWidget(self.run_btn)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Порог уверенности:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        conf_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0.50")
        self.confidence_label.setFixedWidth(40)
        self.confidence_label.setStyleSheet("font-weight: bold; color: #2196f3;")
        conf_layout.addWidget(self.confidence_label)
        control_layout.addLayout(conf_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        right_layout.addWidget(control_group)

        self.results_table = ResultsTableWidget()
        self.results_table.item_selected.connect(self.on_item_selected)
        right_layout.addWidget(self.results_table, stretch=2)

        save_group = QGroupBox("Сохранение")
        save_layout = QHBoxLayout(save_group)

        self.save_full_btn = QPushButton("💾 Сохранить изображение")
        self.save_full_btn.clicked.connect(self.save_full_image)
        self.save_full_btn.setEnabled(False)
        save_layout.addWidget(self.save_full_btn)

        right_layout.addWidget(save_group)

        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([1000, 500])

        main_layout.addWidget(main_splitter)

    def reset_all_zooms(self):
        for container in self.image_containers:
            container.reset_view()
        self.status_bar.showMessage("Зум сброшен во всех окнах", 2000)

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")

        open_action = QAction("Открыть...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("Сохранить изображение...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_full_image)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("Вид")
        reset_zoom_action = QAction("Сбросить зум во всех окнах", self)
        reset_zoom_action.setShortcut("Ctrl+R")
        reset_zoom_action.triggered.connect(self.reset_all_zooms)
        view_menu.addAction(reset_zoom_action)

        help_menu = menubar.addMenu("Помощь")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_tool_bar(self):
        toolbar = QToolBar("Основная")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        open_action = QAction("📂 Открыть", self)
        open_action.triggered.connect(self.open_file_dialog)
        toolbar.addAction(open_action)

        run_action = QAction("▶ Запуск", self)
        run_action.triggered.connect(self.run_analysis)
        toolbar.addAction(run_action)

        toolbar.addSeparator()

        save_action = QAction("💾 Сохранить", self)
        save_action.triggered.connect(self.save_full_image)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        reset_zoom_action = QAction("🔍 Сбросить зум", self)
        reset_zoom_action.triggered.connect(self.reset_all_zooms)
        toolbar.addAction(reset_zoom_action)

    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе. Загрузите изображение.")

    def update_confidence(self, value):
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.clear_memory()

        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение.")
            return

        self.current_image_path = file_path
        self.original_image = img.copy()

        self.display_image_in_container(0, self.original_image)

        self.run_btn.setEnabled(True)
        self.status_bar.showMessage(f"Загружено: {os.path.basename(file_path)}")

    def display_image_in_container(self, index, image):
        if image is None:
            return

        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.image_containers[index].set_image(pixmap)

    def run_analysis(self):
        if self.current_image_path is None:
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        confidence = float(self.confidence_label.text())

        self.processing_thread = ProcessingThread(self.current_image_path, confidence)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.status.connect(self.status_bar.showMessage)
        self.processing_thread.finished.connect(self.on_analysis_finished)
        self.processing_thread.start()

    def on_analysis_finished(self, data):
        if data is None:
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Ошибка обработки изображения")
            return

        self.current_masks = data['masks']
        self.yolo_masks_image = data['yolo_image']
        self.numbered_masks_image = data['numbered_image']
        self.square_list = data.get('square_list', {})

        self.display_image_in_container(1, self.yolo_masks_image)
        self.display_image_in_container(2, self.numbered_masks_image)

        self.image_containers[3].set_image(QPixmap())
        self.image_containers[3].image_label.setText("Выберите сфероид из таблицы")

        self.results_table.update_data(self.current_masks, self.original_image)

        self.run_btn.setEnabled(True)
        self.save_full_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        avg_fill = np.mean([m.get('fill_percentage', 0) for m in self.current_masks]) if self.current_masks else 0
        self.status_bar.showMessage(
            f"Анализ завершен. Найдено {len(self.current_masks)} сфероидов. "
            f"Средняя насыщенность: {avg_fill:.1f}%"
        )

    def on_item_selected(self, idx):
        if idx is not None and idx < len(self.current_masks):
            self.show_selected_spheroid(idx)

    def show_selected_spheroid(self, idx):
        if self.original_image is None:
            return

        mask_data = self.current_masks[idx]
        h, w = self.original_image.shape[:2]
        mask = mask_data['mask']

        try:
            # Получаем маску в правильном размере
            if len(mask.shape) > 2:
                mask = mask.squeeze()

            # Изменяем размер маски до размера изображения
            mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # Находим контуры
            contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                # Если контуры не найдены, используем bbox
                bbox = mask_data.get('bbox')
                if bbox:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    # Создаем прямоугольный контур
                    contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    contours = [contour]
                else:
                    self.status_bar.showMessage(f"Не удалось найти контур для сфероида {mask_data['index']}", 3000)
                    return

            # Выбираем основной контур (самый большой)
            main_contour = max(contours, key=cv2.contourArea)

            # Получаем bounding box для обрезки
            x, y, box_w, box_h = cv2.boundingRect(main_contour)

            # Добавляем отступ
            padding = int(max(box_w, box_h) * 0.3) + 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + box_w + padding)
            y2 = min(h, y + box_h + padding)

            # Проверяем, что область обрезки корректна
            if x1 >= x2 or y1 >= y2:
                x1 = max(0, x - 50)
                y1 = max(0, y - 50)
                x2 = min(w, x + box_w + 50)
                y2 = min(h, y + box_h + 50)

            # Создаем копию области для отображения
            zoom_region = self.original_image[y1:y2, x1:x2].copy()

            # Корректируем контур для обрезанной области
            # adjusted_contour = main_contour - [x1, y1]
            #
            # # Рисуем контур
            # cv2.drawContours(zoom_region, [adjusted_contour], -1, (0, 255, 0), 3)
            #
            # # Рисуем заливку маски с прозрачностью (опционально)
            # mask_cropped = mask_binary[y1:y2, x1:x2]
            # if mask_cropped.shape[0] > 0 and mask_cropped.shape[1] > 0:
            #     # Создаем цветную маску для наложения
            #     colored_mask = np.zeros_like(zoom_region)
            #     colored_mask[mask_cropped == 1] = [255, 100, 0]  # Синий цвет для маски
            #     # Накладываем маску с прозрачностью
            #     zoom_region = cv2.addWeighted(zoom_region, 0.7, colored_mask, 0.3, 0)

            # Рисуем центр и номер
            center = mask_data.get('center')
            if center:
                cx_local = int(center[0] - x1)
                cy_local = int(center[1] - y1)
                radius = mask_data.get('radius', 20)

                # Проверяем, что центр в пределах изображения
                if 0 <= cx_local < zoom_region.shape[1] and 0 <= cy_local < zoom_region.shape[0]:
                    cv2.circle(zoom_region, (cx_local, cy_local), radius + 5, (255, 0, 0), 3)
                    cv2.circle(zoom_region, (cx_local, cy_local), 25, (255, 0, 0), -1)
                    cv2.putText(zoom_region, str(mask_data['index']), (cx_local - 12, cy_local + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Текст для отображения
            fill_pct = mask_data.get('fill_percentage', 0.0)

            # Определяем цвет для насыщенности
            if fill_pct < 40:
                fill_color = (0, 255, 0)  # Зеленый (BGR)
            elif fill_pct < 70:
                fill_color = (0, 165, 255)  # Оранжевый (BGR)
            else:
                fill_color = (0, 0, 255)  # Красный (BGR)

            # Текст для отображения
            info1 = f"ID: {mask_data['index']} | Класс: {mask_data['class_name']} | Уверенность: {mask_data['conf']:.2f}"
            info2 = f"Насыщенность: {fill_pct:.1f}%"

            # Рисуем русский текст с помощью PIL
            try:
                zoom_region = draw_russian_text(zoom_region, info1, (10, 30), font_size=16,
                                                color=(255, 255, 255), outline_color=(0, 0, 0), outline_width=2)
                zoom_region = draw_russian_text(zoom_region, info2, (10, 60), font_size=16,
                                                color=fill_color, outline_color=(0, 0, 0), outline_width=2)
            except Exception as e:
                # Если не удалось нарисовать русский текст, используем английский
                print(f"Error drawing Russian text: {e}")
                info1_en = f"ID: {mask_data['index']} | Class: {mask_data['class_name']} | Conf: {mask_data['conf']:.2f}"
                info2_en = f"Fill rate: {fill_pct:.1f}%"
                cv2.putText(zoom_region, info1_en, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(zoom_region, info2_en, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fill_color, 2)

            # Рисуем рамку
            cv2.rectangle(zoom_region, (0, 0), (zoom_region.shape[1] - 1, zoom_region.shape[0] - 1), (0, 255, 0), 3)

            # Дополнительная информация о площади
            area = cv2.contourArea(main_contour)
            info3 = f"Площадь: {area:.0f} пикс."
            try:
                zoom_region = draw_russian_text(zoom_region, info3, (10, 85), font_size=12,
                                                color=(200, 200, 200), outline_color=(0, 0, 0), outline_width=1)
            except:
                cv2.putText(zoom_region, f"Area: {area:.0f} px", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (200, 200, 200), 1)

            # Конвертируем и отображаем
            rgb_zoom = cv2.cvtColor(zoom_region, cv2.COLOR_BGR2RGB)
            h_z, w_z, ch_z = rgb_zoom.shape
            bytes_per_line = ch_z * w_z
            qt_image = QImage(rgb_zoom.data, w_z, h_z, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            self.image_containers[3].set_image(pixmap)

        except Exception as e:
            self.status_bar.showMessage(f"Ошибка при отображении сфероида: {str(e)}", 3000)
            print(f"Error in show_selected_spheroid: {e}")
            import traceback
            traceback.print_exc()

    def save_full_image(self):
        if self.numbered_masks_image is None:
            QMessageBox.warning(self, "Ошибка", "Нет результатов для сохранения.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "result.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if file_path:
            cv2.imwrite(file_path, self.numbered_masks_image)
            self.status_bar.showMessage(f"Изображение сохранено: {file_path}")

    def clear_memory(self):
        self.current_image_path = None
        self.current_masks = []
        self.original_image = None
        self.yolo_masks_image = None
        self.numbered_masks_image = None
        self.square_list = {}

        for container in self.image_containers:
            container.clear()
        self.results_table.clear()

        self.image_containers[3].image_label.setText("Выберите сфероид из таблицы")

        torch.cuda.empty_cache()
        gc.collect()

    def show_about(self):
        QMessageBox.about(self, "О программе", """
            <h3>Spheroid Segmentation with YOLO</h3>
            <p>Версия: 2.0</p>
            <p>Программа для сегментации и анализа сфероидов на изображениях с микроскопа.</p>
            <p><b>Особенности:</b></p>
            <ul>
                <li>Автоматическое обнаружение сфероидов с помощью YOLO</li>
                <li>Фильтрация дубликатов и сфероидов на границах</li>
                <li>Расчет насыщенности (процент белых пикселей внутри сфероида)</li>
                <li>Интерактивная таблица с результатами</li>
                <li>Зум и панорамирование изображений</li>
            </ul>
            <p><b>Управление изображениями:</b></p>
            <ul>
                <li><b>Колесико мыши</b> - приближение/отдаление (от 0.5x до 3x)</li>
                <li><b>Левая кнопка мыши + перетаскивание</b> - панорамирование</li>
                <li><b>Ctrl+R</b> - сброс зума во всех окнах</li>
            </ul>
            <p><b>Насыщенность:</b></p>
            <ul>
                <li>🟢 Зеленый - низкая (&lt;40%)</li>
                <li>🟠 Оранжевый - средняя (40-70%)</li>
                <li>🔴 Красный - высокая (&gt;70%)</li>
            </ul>
        """)

    def closeEvent(self, event):
        self.clear_memory()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()