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

# Загрузка модели YOLO
model = YOLO("best.pt")


class ProcessingThread(QThread):
    """Поток для обработки изображения"""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, image, confidence_threshold=0.5):
        super().__init__()
        self.image = image
        self.confidence_threshold = confidence_threshold

    def run(self):
        self.status.emit("Запуск модели YOLO...")
        self.progress.emit(20)

        with torch.no_grad():
            result = model(self.image)[0]

        self.progress.emit(50)
        self.status.emit("Обработка результатов...")

        masks_data = []

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else np.zeros(len(boxes))

            for i, mask in enumerate(masks):
                conf = float(boxes[i].conf)
                if conf >= self.confidence_threshold:
                    class_idx = int(classes[i])
                    class_name = result.names[class_idx] if hasattr(result, 'names') else f"Class {class_idx}"

                    if hasattr(result.boxes, 'xyxy'):
                        bbox = result.boxes.xyxy[i].cpu().numpy()
                    else:
                        bbox = [0, 0, 0, 0]

                    masks_data.append({
                        'mask': mask,
                        'conf': conf,
                        'index': i,
                        'class_name': class_name,
                        'class_idx': class_idx,
                        'bbox': bbox
                    })

        self.progress.emit(80)
        self.status.emit("Визуализация...")

        plotted = result.plot()

        self.progress.emit(100)
        self.status.emit("Готово!")

        self.finished.emit({
            'result': result,
            'masks': masks_data,
            'plotted': plotted,
            'original_shape': self.image.shape
        })


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
        """Установка изображения"""
        self.original_pixmap = pixmap
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()

    def update_display(self):
        """Обновление отображения с учетом зума и панорамирования"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            self.clear()
            self.setText("Нет изображения")
            return

        try:
            if self.zoom <= 1.0:
                # При зуме <= 1 просто масштабируем под размер
                scaled = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled)
                self.current_pixmap = scaled
            else:
                # При зуме > 1 создаем увеличенную версию
                orig_w = self.original_pixmap.width()
                orig_h = self.original_pixmap.height()

                new_w = int(orig_w * self.zoom)
                new_h = int(orig_h * self.zoom)

                # Ограничиваем максимальный размер
                max_size = 3000
                if new_w > max_size or new_h > max_size:
                    scale = min(max_size / new_w, max_size / new_h)
                    new_w = int(new_w * scale)
                    new_h = int(new_h * scale)
                    self.zoom = new_w / orig_w

                scaled = self.original_pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Получаем размеры виджета
                widget_w = self.width()
                widget_h = self.height()

                # Вычисляем видимую область
                x = self.pan_x
                y = self.pan_y

                # Ограничиваем панорамирование
                max_pan_x = max(0, scaled.width() - widget_w)
                max_pan_y = max(0, scaled.height() - widget_h)
                x = max(0, min(x, max_pan_x))
                y = max(0, min(y, max_pan_y))

                # Обновляем pan_x, pan_y
                self.pan_x = x
                self.pan_y = y

                # Вырезаем область
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
        """При изменении размера виджета обновляем отображение"""
        self.update_display()
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Обработка колесика мыши для зума"""
        if self.original_pixmap is None:
            return

        # Сохраняем текущий зум
        old_zoom = self.zoom

        # Изменяем зум
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom *= 0.9

        # Ограничиваем зум
        self.zoom = max(0.5, min(self.zoom, 3.0))

        # Если зум изменился, корректируем панорамирование
        if self.zoom != old_zoom and self.zoom > 1.0:
            zoom_ratio = self.zoom / old_zoom
            self.pan_x = int(self.pan_x * zoom_ratio)
            self.pan_y = int(self.pan_y * zoom_ratio)

        self.update_display()

    def mousePressEvent(self, event):
        """Начало панорамирования"""
        if event.button() == Qt.LeftButton and self.zoom > 1.0:
            self.drag_start = event.pos()
            self.last_mouse_pos = event.pos()
            self.is_dragging = True
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Панорамирование при зажатой левой кнопке"""
        if self.is_dragging and self.drag_start and self.zoom > 1.0:
            if self.last_mouse_pos:
                delta = event.pos() - self.last_mouse_pos
                self.pan_x -= delta.x()
                self.pan_y -= delta.y()

                # Ограничиваем панорамирование
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
        """Завершение панорамирования"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.drag_start = None
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def clear(self):
        """Очистка изображения"""
        self.original_pixmap = None
        self.current_pixmap = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        super().clear()
        self.setText("Нет изображения")

    def reset_view(self):
        """Сброс зума и панорамирования"""
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

        # Создаем область прокрутки
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: #2b2b2b; }")

        # Виджет с изображением
        self.image_label = ZoomableImageLabel()
        self.scroll_area.setWidget(self.image_label)

        layout.addWidget(self.scroll_area)

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
    """Виджет с фиксированной таблицей результатов"""

    item_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Заголовок
        self.header_label = QLabel("📊 Результаты сегментации")
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

        # Создание таблицы
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "№", "Класс", "Уверенность", "Площадь", "Радиус",
            "Периметр", "Компактность", "Центр X", "Центр Y"
        ])

        # Настройка внешнего вида таблицы
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSortingEnabled(True)

        # Запрещаем изменение размеров столбцов
        header = self.table.horizontalHeader()
        header.setSectionsMovable(False)
        header.setSectionResizeMode(QHeaderView.Fixed)

        # Фиксированная ширина столбцов
        column_widths = [40, 100, 90, 80, 70, 80, 100, 80, 80]
        for i, width in enumerate(column_widths):
            self.table.setColumnWidth(i, width)

        # Шрифт
        font = QFont()
        font.setPointSize(10)
        self.table.setFont(font)

        # Подключение сигнала
        self.table.itemSelectionChanged.connect(self.on_selection_changed)

        layout.addWidget(self.table)

        # Панель статистики
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

        self.avg_area_label = QLabel("Ср. площадь: 0")
        self.avg_area_label.setStyleSheet("padding: 5px;")
        stats_layout.addWidget(self.avg_area_label)

        self.avg_confidence_label = QLabel("Ср. уверенность: 0")
        self.avg_confidence_label.setStyleSheet("padding: 5px;")
        stats_layout.addWidget(self.avg_confidence_label)

        layout.addWidget(self.stats_frame)

        # Кнопка сброса зума
        reset_zoom_btn = QPushButton("🔄 Сбросить зум во всех окнах")
        reset_zoom_btn.clicked.connect(self.reset_all_zooms)
        layout.addWidget(reset_zoom_btn)

    def reset_all_zooms(self):
        """Сброс зума во всех окнах"""
        main_window = self.window()
        if hasattr(main_window, 'reset_all_zooms'):
            main_window.reset_all_zooms()

    def update_data(self, masks_data, original_image=None):
        """Обновление данных в таблице"""
        self.table.setRowCount(0)

        if not masks_data:
            self.update_statistics(0, [], [])
            return

        h, w = original_image.shape[:2] if original_image is not None else (1, 1)

        areas = []
        confidences = []

        for i, mask_data in enumerate(masks_data):
            row = self.table.rowCount()
            self.table.insertRow(row)

            area = self.calculate_mask_area(mask_data['mask'], w, h)
            radius = np.sqrt(area / np.pi)
            perimeter = self.calculate_mask_perimeter(mask_data['mask'], w, h)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            center_x, center_y = self.calculate_center(mask_data['mask'], w, h)

            areas.append(area)
            confidences.append(mask_data['conf'])

            self.table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(mask_data['class_name']))

            conf_item = QTableWidgetItem(f"{mask_data['conf']:.3f}")
            if mask_data['conf'] > 0.7:
                conf_item.setForeground(QBrush(QColor(76, 175, 80)))
            elif mask_data['conf'] > 0.4:
                conf_item.setForeground(QBrush(QColor(255, 152, 0)))
            else:
                conf_item.setForeground(QBrush(QColor(244, 67, 54)))
            self.table.setItem(row, 2, conf_item)

            self.table.setItem(row, 3, QTableWidgetItem(f"{area:.0f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{radius:.1f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{perimeter:.1f}"))

            compact_item = QTableWidgetItem(f"{compactness:.3f}")
            if compactness > 0.8:
                compact_item.setForeground(QBrush(QColor(76, 175, 80)))
            elif compactness > 0.5:
                compact_item.setForeground(QBrush(QColor(255, 152, 0)))
            else:
                compact_item.setForeground(QBrush(QColor(244, 67, 54)))
            self.table.setItem(row, 6, compact_item)

            self.table.setItem(row, 7, QTableWidgetItem(f"{center_x:.1f}"))
            self.table.setItem(row, 8, QTableWidgetItem(f"{center_y:.1f}"))

            self.table.item(row, 0).setData(Qt.UserRole, i)

        self.update_statistics(len(masks_data), areas, confidences)
        self.table.resizeRowsToContents()

    def calculate_mask_area(self, mask, img_w, img_h):
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return float(np.sum(mask_resized > 0.5))

    def calculate_mask_perimeter(self, mask, img_w, img_h):
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return float(cv2.arcLength(contours[0], True))
        return 0.0

    def calculate_center(self, mask, img_w, img_h):
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        moments = cv2.moments(mask_binary)
        if moments["m00"] != 0:
            return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        return 0, 0

    def update_statistics(self, total_count, areas, confidences):
        self.total_count_label.setText(f"📈 Всего объектов: {total_count}")
        if areas:
            self.avg_area_label.setText(f"📐 Ср. площадь: {np.mean(areas):.0f} px²")
        else:
            self.avg_area_label.setText("📐 Ср. площадь: 0 px²")
        if confidences:
            self.avg_confidence_label.setText(f"🎯 Ср. уверенность: {np.mean(confidences):.3f}")
        else:
            self.avg_confidence_label.setText("🎯 Ср. уверенность: 0")

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

        self.current_image = None
        self.current_result = None
        self.current_masks = []
        self.original_image = None
        self.yolo_masks_image = None
        self.numbered_masks_image = None

        self.setup_ui()
        self.create_menu_bar()
        self.create_tool_bar()
        self.setup_status_bar()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # ===== ВЕРХНЯЯ ПАНЕЛЬ С ИНФОРМАЦИЕЙ О ЗУМЕ =====
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        info_layout = QHBoxLayout(info_frame)

        info_icon = QLabel("🖱️")
        info_icon.setStyleSheet("font-size: 20px;")
        info_layout.addWidget(info_icon)

        info_text = QLabel(
            "Управление изображениями: Колесико мыши - приближение/отдаление (0.5x - 3x) | Левая кнопка + перетаскивание - панорамирование (при зуме > 1x) | Ctrl+R - сброс зума")
        info_text.setStyleSheet("font-size: 12px; color: #333; padding: 5px;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text, 1)

        main_layout.addWidget(info_frame)

        # ===== ОСНОВНОЙ СПЛИТТЕР =====
        main_splitter = QSplitter(Qt.Horizontal)

        # ===== ЛЕВАЯ ПАНЕЛЬ - Изображения (4 окна) =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Сетка изображений 2x2
        images_container = QWidget()
        self.image_grid = QGridLayout(images_container)
        self.image_grid.setSpacing(10)

        self.image_containers = []
        titles = ["Оригинал", "Маски YOLO", "Пронумерованные маски", "Выбранный сфероид"]

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

            # Контейнер с изображением
            img_container = ImageContainer()
            layout.addWidget(img_container)

            row, col = i // 2, i % 2
            self.image_grid.addWidget(frame, row, col)
            self.image_containers.append(img_container)

        # Очищаем 4-е окно (индекс 3)
        self.image_containers[3].set_image(QPixmap())
        self.image_containers[3].image_label.setText("Выберите сфероид из таблицы")

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(images_container)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        left_layout.addWidget(scroll_area)

        # ===== ПРАВАЯ ПАНЕЛЬ - Управление =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Drop zone для загрузки файлов
        self.drop_zone = DropZone("📁 Перетащите изображение сюда\nили используйте кнопку ниже")
        self.drop_zone.file_dropped.connect(self.load_image)
        right_layout.addWidget(self.drop_zone)

        # Кнопка загрузки через проводник
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

        # Группа управления моделью
        control_group = QGroupBox("Управление моделью")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        control_layout = QVBoxLayout(control_group)

        self.run_btn = QPushButton("▶ Запустить сегментацию")
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
        self.run_btn.clicked.connect(self.run_segmentation)
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

        self.apply_threshold_btn = QPushButton("🔄 Применить порог")
        self.apply_threshold_btn.clicked.connect(self.apply_threshold)
        self.apply_threshold_btn.setEnabled(False)
        control_layout.addWidget(self.apply_threshold_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        right_layout.addWidget(control_group)

        # Таблица результатов
        self.results_table = ResultsTableWidget()
        self.results_table.item_selected.connect(self.on_item_selected)
        right_layout.addWidget(self.results_table, stretch=2)

        # Сохранение
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
        """Сброс зума во всех окнах"""
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
        run_action.triggered.connect(self.run_segmentation)
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

        self.original_image = cv2.resize(img, (1024, 1024))
        self.display_image_in_container(0, self.original_image)
        self.current_image = self.original_image.copy()

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

    def create_numbered_masks_image(self):
        """Создание изображения с пронумерованными масками"""
        if self.original_image is None or not self.current_masks:
            return

        h, w = self.original_image.shape[:2]
        self.numbered_masks_image = self.original_image.copy()

        for i, mask_data in enumerate(self.current_masks):
            mask = mask_data['mask']
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(self.numbered_masks_image, contours, -1, (0, 255, 0), 2)

            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(self.numbered_masks_image, (cx, cy), 15, (255, 0, 0), -1)
                    cv2.putText(self.numbered_masks_image, str(i + 1), (cx - 7, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def run_segmentation(self):
        if self.current_image is None:
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        confidence = float(self.confidence_label.text())

        self.processing_thread = ProcessingThread(self.current_image, confidence)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.status.connect(self.status_bar.showMessage)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self, data):
        self.current_result = data['result']
        self.current_masks = data['masks']
        self.yolo_masks_image = data['plotted']

        self.create_numbered_masks_image()

        self.display_image_in_container(1, self.yolo_masks_image)
        self.display_image_in_container(2, self.numbered_masks_image)

        # Очищаем 4-е окно
        self.image_containers[3].set_image(QPixmap())
        self.image_containers[3].image_label.setText("Выберите сфероид из таблицы")

        self.results_table.update_data(self.current_masks, self.original_image)

        self.run_btn.setEnabled(True)
        self.apply_threshold_btn.setEnabled(True)
        self.save_full_btn.setEnabled(True)

        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Сегментация завершена. Найдено {len(self.current_masks)} объектов")

    def apply_threshold(self):
        if self.current_result is None:
            return

        confidence = float(self.confidence_label.text())
        filtered_masks = [m for m in self.current_masks if m['conf'] >= confidence]
        self.current_masks = filtered_masks

        self.processing_thread = ProcessingThread(self.current_image, confidence)
        self.processing_thread.finished.connect(self.on_threshold_applied)
        self.processing_thread.start()

    def on_threshold_applied(self, data):
        self.current_result = data['result']
        self.current_masks = data['masks']
        self.yolo_masks_image = data['plotted']

        self.create_numbered_masks_image()

        self.display_image_in_container(1, self.yolo_masks_image)
        self.display_image_in_container(2, self.numbered_masks_image)

        # Очищаем 4-е окно
        self.image_containers[3].set_image(QPixmap())
        self.image_containers[3].image_label.setText("Выберите сфероид из таблицы")

        self.results_table.update_data(self.current_masks, self.original_image)
        self.status_bar.showMessage(f"Применен порог, осталось {len(self.current_masks)} объектов")

    def on_item_selected(self, idx):
        """Обработка выбора элемента из таблицы - показываем выбранный сфероид в 4-м окне"""
        if idx is not None and idx < len(self.current_masks):
            self.show_selected_spheroid(idx)

    def show_selected_spheroid(self, idx):
        """Показ выбранного сфероида в 4-м окне"""
        if self.original_image is None:
            return

        mask_data = self.current_masks[idx]
        h, w = self.original_image.shape[:2]
        mask = mask_data['mask']

        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, box_w, box_h = cv2.boundingRect(contours[0])

            # Добавляем отступ 50%
            padding = int(max(box_w, box_h) * 0.5)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + box_w + padding)
            y2 = min(h, y + box_h + padding)

            # Вырезаем область из оригинального изображения
            zoom_region = self.original_image[y1:y2, x1:x2].copy()

            # Рисуем контур зеленым
            for cnt in contours:
                cnt_adjusted = cnt - [x1, y1]
                cv2.drawContours(zoom_region, [cnt_adjusted], -1, (0, 255, 0), 3)

            # Добавляем номер
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) - x1
                    cy = int(M["m01"] / M["m00"]) - y1
                    cv2.circle(zoom_region, (cx, cy), 20, (255, 0, 0), -1)
                    cv2.putText(zoom_region, str(idx + 1), (cx - 8, cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    break

            # Добавляем информацию
            info = f"ID: {idx + 1} | {mask_data['class_name']} | conf: {mask_data['conf']:.2f}"
            cv2.putText(zoom_region, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Добавляем рамку
            cv2.rectangle(zoom_region, (0, 0), (zoom_region.shape[1] - 1, zoom_region.shape[0] - 1), (0, 255, 0), 3)

            # Отображаем
            rgb_zoom = cv2.cvtColor(zoom_region, cv2.COLOR_BGR2RGB)
            h_z, w_z, ch_z = rgb_zoom.shape
            bytes_per_line = ch_z * w_z
            qt_image = QImage(rgb_zoom.data, w_z, h_z, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            self.image_containers[3].set_image(pixmap)

    def save_full_image(self):
        if self.current_result is None:
            QMessageBox.warning(self, "Ошибка", "Нет результатов для сохранения.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "result.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if file_path:
            cv2.imwrite(file_path, self.current_result.plot())
            self.status_bar.showMessage(f"Изображение сохранено: {file_path}")

    def clear_memory(self):
        self.current_image = None
        self.current_result = None
        self.current_masks = []
        self.yolo_masks_image = None
        self.numbered_masks_image = None

        for container in self.image_containers:
            container.clear()
        self.results_table.clear()

        # Устанавливаем текст для 4-го окна
        self.image_containers[3].image_label.setText("Выберите сфероид из таблицы")

        torch.cuda.empty_cache()
        gc.collect()

    def show_about(self):
        QMessageBox.about(self, "О программе", """
            <h3>Spheroid Segmentation with YOLO</h3>
            <p>Версия: 1.0</p>
            <p>Программа для сегментации сфероидов на изображениях с микроскопа.</p>
            <p><b>Управление изображениями:</b></p>
            <ul>
                <li><b>Колесико мыши</b> - приближение/отдаление (от 0.5x до 3x)</li>
                <li><b>Левая кнопка мыши + перетаскивание</b> - панорамирование (работает при любом зуме > 1x)</li>
                <li><b>Ctrl+R</b> - сброс зума во всех окнах</li>
            </ul>
            <p><b>Окна:</b></p>
            <ul>
                <li><b>Оригинал</b> - исходное изображение</li>
                <li><b>Маски YOLO</b> - результат работы YOLO</li>
                <li><b>Пронумерованные маски</b> - зеленые контуры с номерами</li>
                <li><b>Выбранный сфероид</b> - увеличенное изображение выбранного сфероида</li>
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