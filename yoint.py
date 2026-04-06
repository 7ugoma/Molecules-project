import sys
import os
import cv2
import numpy as np
import torch
import gc
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QFileDialog, QMessageBox, QSlider, QSplitter,
                             QTreeWidget, QTreeWidgetItem, QFrame, QProgressBar,
                             QStatusBar, QToolBar, QAction, QGroupBox, QHeaderView,
                             QTableWidget, QTableWidgetItem, QTabWidget, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from ultralytics import YOLO

# Загрузка модели YOLO
model = YOLO("best.pt")


class ProcessingThread(QThread):
    """Поток для обработки изображения, чтобы не блокировать интерфейс"""
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

        # Извлечение масок и боксов
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

                    # Получение координат bounding box
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

        # Создание визуализации
        plotted = result.plot()

        self.progress.emit(100)
        self.status.emit("Готово!")

        self.finished.emit({
            'result': result,
            'masks': masks_data,
            'plotted': plotted,
            'original_shape': self.image.shape
        })


class ScrollableImageLabel(QWidget):
    """Виджет с прокруткой для просмотра изображения"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_pixmap = None
        self.zoom = 1.0

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Создание области прокрутки
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #ccc;
                border-radius: 5px;
                background: #2b2b2b;
            }
        """)

        # Виджет для отображения изображения
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #2b2b2b;")

        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

    def set_image(self, pixmap):
        """Установка изображения с сохранением пропорций"""
        self.current_pixmap = pixmap
        self.update_display()

    def update_display(self):
        """Обновление отображения с учетом зума"""
        if self.current_pixmap is None:
            return

        # Масштабирование с учетом зума
        w = int(self.current_pixmap.width() * self.zoom)
        h = int(self.current_pixmap.height() * self.zoom)
        scaled_pixmap = self.current_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        """Обработка колесика мыши для зума"""
        if self.current_pixmap is None:
            return

        # Сохраняем позицию до зума
        old_pos = self.scroll_area.mapFromGlobal(event.globalPos())

        # Изменяем зум
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.2, min(self.zoom, 10))

        # Обновляем изображение
        self.update_display()

        # Корректируем позицию прокрутки чтобы центр оставался на месте
        new_pos = self.scroll_area.mapFromGlobal(event.globalPos())
        delta = new_pos - old_pos
        self.scroll_area.horizontalScrollBar().setValue(
            self.scroll_area.horizontalScrollBar().value() + delta.x()
        )
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().value() + delta.y()
        )

    def clear(self):
        """Очистка изображения"""
        self.current_pixmap = None
        self.image_label.clear()
        self.zoom = 1.0


class DropZone(QLabel):
    """Область для Drag & Drop файлов"""

    file_dropped = pyqtSignal(str)

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(150)
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
    """Виджет с увеличенной таблицей результатов"""

    item_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Заголовок с подсчетом
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
            "№",
            "Класс",
            "Уверенность",
            "Площадь (px²)",
            "Радиус (px)",
            "Периметр (px)",
            "Компактность",
            "Центр X",
            "Центр Y"
        ])

        # Настройка внешнего вида таблицы
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSortingEnabled(True)

        # Настройка шрифтов и размеров
        font = QFont()
        font.setPointSize(11)
        self.table.setFont(font)

        # Растягивание колонок
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)

        # Установка минимальных ширин колонок
        self.table.setColumnWidth(0, 50)  # №
        self.table.setColumnWidth(1, 120)  # Класс
        self.table.setColumnWidth(2, 100)  # Уверенность
        self.table.setColumnWidth(3, 120)  # Площадь
        self.table.setColumnWidth(4, 100)  # Радиус
        self.table.setColumnWidth(5, 100)  # Периметр
        self.table.setColumnWidth(6, 120)  # Компактность
        self.table.setColumnWidth(7, 100)  # Центр X
        self.table.setColumnWidth(8, 100)  # Центр Y

        # Включение сортировки
        self.table.setSortingEnabled(True)

        # Подключение сигнала выбора
        self.table.itemSelectionChanged.connect(self.on_selection_changed)

        layout.addWidget(self.table)

        # Панель статистики внизу таблицы
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

            # Расчет параметров
            area = self.calculate_mask_area(mask_data['mask'], w, h)
            radius = np.sqrt(area / np.pi)
            perimeter = self.calculate_mask_perimeter(mask_data['mask'], w, h)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            # Центр масс
            center_x, center_y = self.calculate_center(mask_data['mask'], w, h)

            areas.append(area)
            confidences.append(mask_data['conf'])

            # Заполнение ячеек
            self.table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(mask_data['class_name']))

            conf_item = QTableWidgetItem(f"{mask_data['conf']:.3f}")
            if mask_data['conf'] > 0.7:
                conf_item.setForeground(QBrush(QColor(76, 175, 80)))  # Зеленый для высокой уверенности
            elif mask_data['conf'] > 0.4:
                conf_item.setForeground(QBrush(QColor(255, 152, 0)))  # Оранжевый для средней
            else:
                conf_item.setForeground(QBrush(QColor(244, 67, 54)))  # Красный для низкой
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

            # Сохраняем индекс в данных строки
            self.table.item(row, 0).setData(Qt.UserRole, i)

        # Обновление статистики
        self.update_statistics(len(masks_data), areas, confidences)

        # Настройка высоты строк
        self.table.resizeRowsToContents()

    def calculate_mask_area(self, mask, img_w, img_h):
        """Расчет площади маски в пикселях"""
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return float(np.sum(mask_resized > 0.5))

    def calculate_mask_perimeter(self, mask, img_w, img_h):
        """Расчет периметра маски"""
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return float(cv2.arcLength(contours[0], True))
        return 0.0

    def calculate_center(self, mask, img_w, img_h):
        """Расчет центра масс"""
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        moments = cv2.moments(mask_binary)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return cx, cy
        return 0, 0

    def update_statistics(self, total_count, areas, confidences):
        """Обновление статистической информации"""
        self.total_count_label.setText(f"📈 Всего объектов: {total_count}")

        if areas:
            avg_area = np.mean(areas)
            self.avg_area_label.setText(f"📐 Ср. площадь: {avg_area:.0f} px²")
        else:
            self.avg_area_label.setText("📐 Ср. площадь: 0 px²")

        if confidences:
            avg_conf = np.mean(confidences)
            self.avg_confidence_label.setText(f"🎯 Ср. уверенность: {avg_conf:.3f}")
        else:
            self.avg_confidence_label.setText("🎯 Ср. уверенность: 0")

    def on_selection_changed(self):
        """Обработка изменения выбора в таблице"""
        selected_rows = self.table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            item = self.table.item(row, 0)
            if item:
                idx = item.data(Qt.UserRole)
                self.item_selected.emit(idx)

    def clear(self):
        """Очистка таблицы"""
        self.table.setRowCount(0)
        self.update_statistics(0, [], [])


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spheroid Segmentation with YOLO - PyQt Interface")
        self.setGeometry(100, 100, 1600, 1000)

        self.current_image = None
        self.current_result = None
        self.current_masks = []
        self.original_image = None

        self.setup_ui()
        self.create_menu_bar()
        self.create_tool_bar()
        self.setup_status_bar()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной сплиттер
        main_splitter = QSplitter(Qt.Horizontal)

        # ===== ЛЕВАЯ ПАНЕЛЬ - Изображения =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Drop zone для загрузки
        self.drop_zone = DropZone("📁 Перетащите изображение сюда\nили используйте меню Файл → Открыть")
        self.drop_zone.file_dropped.connect(self.load_image)
        left_layout.addWidget(self.drop_zone)

        # Область для отображения изображений (сетка 2x2 с прокруткой)
        self.image_grid = QGridLayout()
        self.image_labels = []
        self.image_titles = []

        titles = ["Оригинал", "Все контуры", "Похожие контуры", "Результат"]

        for i in range(4):
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 5px; }")
            frame.setFixedSize(550, 450)  # Фиксированный размер окна

            layout = QVBoxLayout(frame)
            layout.setContentsMargins(5, 5, 5, 5)

            title_label = QLabel(titles[i])
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px; background: #f0f0f0; padding: 5px;")
            title_label.setFixedHeight(30)

            # Используем виджет с прокруткой
            img_label = ScrollableImageLabel()

            layout.addWidget(title_label)
            layout.addWidget(img_label)

            row, col = i // 2, i % 2
            self.image_grid.addWidget(frame, row, col)
            self.image_labels.append(img_label)
            self.image_titles.append(title_label)

        # Контейнер для сетки с прокруткой
        grid_container = QScrollArea()
        grid_container.setWidgetResizable(True)
        grid_container.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grid_container.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grid_container.setStyleSheet("QScrollArea { border: none; }")

        grid_widget = QWidget()
        grid_widget.setLayout(self.image_grid)
        grid_container.setWidget(grid_widget)

        left_layout.addWidget(grid_container)

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
            QPushButton:hover {
                background-color: #b3e5fc;
            }
        """)
        self.upload_btn.clicked.connect(self.open_file_dialog)
        left_layout.addWidget(self.upload_btn)

        # ===== ПРАВАЯ ПАНЕЛЬ - Управление и результаты =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Группа управления моделью
        control_group = QGroupBox("Управление моделью")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        control_layout = QVBoxLayout(control_group)

        # Кнопка запуска
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
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self.run_segmentation)
        self.run_btn.setEnabled(False)
        control_layout.addWidget(self.run_btn)

        # Ползунок уверенности
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Порог уверенности:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        confidence_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0.50")
        self.confidence_label.setFixedWidth(40)
        self.confidence_label.setStyleSheet("font-weight: bold; color: #2196f3;")
        confidence_layout.addWidget(self.confidence_label)
        control_layout.addLayout(confidence_layout)

        # Кнопка применения порога
        self.apply_threshold_btn = QPushButton("🔄 Применить порог")
        self.apply_threshold_btn.clicked.connect(self.apply_threshold)
        self.apply_threshold_btn.setEnabled(False)
        control_layout.addWidget(self.apply_threshold_btn)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        right_layout.addWidget(control_group)

        # Таблица результатов
        self.results_table = ResultsTableWidget()
        self.results_table.item_selected.connect(self.on_item_selected)
        right_layout.addWidget(self.results_table, stretch=2)

        # Кнопки сохранения
        save_group = QGroupBox("Экспорт данных")
        save_layout = QHBoxLayout(save_group)

        self.save_full_btn = QPushButton("💾 Сохранить изображение")
        self.save_full_btn.clicked.connect(self.save_full_image)
        self.save_full_btn.setEnabled(False)
        save_layout.addWidget(self.save_full_btn)

        self.save_csv_btn = QPushButton("📊 Сохранить CSV")
        self.save_csv_btn.clicked.connect(self.save_csv)
        self.save_csv_btn.setEnabled(False)
        save_layout.addWidget(self.save_csv_btn)

        right_layout.addWidget(save_group)

        # Добавляем виджеты в сплиттер
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([1200, 500])

        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(main_splitter)

    def create_menu_bar(self):
        """Создание меню"""
        menubar = self.menuBar()

        # Файл
        file_menu = menubar.addMenu("Файл")

        open_action = QAction("Открыть...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_full_action = QAction("Сохранить изображение...", self)
        save_full_action.setShortcut("Ctrl+S")
        save_full_action.triggered.connect(self.save_full_image)
        file_menu.addAction(save_full_action)

        save_csv_action = QAction("Сохранить CSV...", self)
        save_csv_action.setShortcut("Ctrl+Shift+S")
        save_csv_action.triggered.connect(self.save_csv)
        file_menu.addAction(save_csv_action)

        file_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Помощь
        help_menu = menubar.addMenu("Помощь")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_tool_bar(self):
        """Создание панели инструментов"""
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

    def setup_status_bar(self):
        """Настройка статусной строки"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе. Загрузите изображение.")

    def update_confidence(self, value):
        """Обновление значения уверенности"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")

    def open_file_dialog(self):
        """Открытие диалога выбора файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """Загрузка изображения из файла"""
        # Очистка памяти
        self.clear_memory()

        # Загрузка изображения
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение.")
            return

        self.original_image = cv2.resize(img, (1024, 1024))

        # Отображение в интерфейсе
        self.display_image_in_label(0, self.original_image)

        self.current_image = self.original_image.copy()

        # Активация кнопок
        self.run_btn.setEnabled(True)
        self.status_bar.showMessage(f"Загружено: {os.path.basename(file_path)}")

    def display_image_in_label(self, index, image):
        """Отображение изображения в указанной метке"""
        if image is None:
            return

        # Конвертация BGR (OpenCV) в RGB для Qt
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.image_labels[index].set_image(pixmap)

    def run_segmentation(self):
        """Запуск сегментации в отдельном потоке"""
        if self.current_image is None:
            return

        # Блокировка кнопок во время обработки
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        confidence = float(self.confidence_label.text())

        # Запуск потока обработки
        self.processing_thread = ProcessingThread(self.current_image, confidence)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.status.connect(self.status_bar.showMessage)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self, data):
        """Обработка завершения сегментации"""
        self.current_result = data['result']
        self.current_masks = data['masks']

        # Отображение результатов
        plotted = data['plotted']
        self.display_image_in_label(3, plotted)

        # Создание изображений для контуров
        self.create_contour_visualizations()

        # Заполнение таблицы результатов
        self.results_table.update_data(self.current_masks, self.original_image)

        # Активация кнопок
        self.run_btn.setEnabled(True)
        self.apply_threshold_btn.setEnabled(True)
        self.save_full_btn.setEnabled(True)
        self.save_csv_btn.setEnabled(True)

        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Сегментация завершена. Найдено {len(self.current_masks)} объектов")

    def create_contour_visualizations(self):
        """Создание визуализаций для всех контуров и похожих контуров"""
        if self.original_image is None or not self.current_masks:
            return

        h, w = self.original_image.shape[:2]

        # Изображение для всех контуров
        all_contours_img = self.original_image.copy()
        # Изображение для похожих контуров (по площади)
        similar_contours_img = self.original_image.copy()

        # Находим среднюю площадь для определения "похожих"
        areas = []
        for mask_data in self.current_masks:
            area = self.calculate_mask_area(mask_data['mask'], w, h)
            areas.append(area)

        if areas:
            avg_area = np.mean(areas)
            tolerance = 0.3  # 30% допуск

        for i, mask_data in enumerate(self.current_masks):
            mask = mask_data['mask']

            # Масштабирование маски до размера изображения
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # Нахождение контуров
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Рисуем все контуры зеленым
            cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)

            # Для похожих контуров
            area = areas[i] if i < len(areas) else 0
            if areas and abs(area - avg_area) / avg_area <= tolerance:
                cv2.drawContours(similar_contours_img, contours, -1, (0, 255, 0), 2)
                # Добавляем номер
                M = cv2.moments(contours[0]) if contours else None
                if M and M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(similar_contours_img, str(i + 1), (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Отображение
        self.display_image_in_label(1, all_contours_img)
        self.display_image_in_label(2, similar_contours_img)

    def calculate_mask_area(self, mask, img_w, img_h):
        """Расчет площади маски в пикселях"""
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return float(np.sum(mask_resized > 0.5))

    def apply_threshold(self):
        """Применение порога уверенности к результатам"""
        if self.current_result is None:
            return

        confidence = float(self.confidence_label.text())

        # Фильтрация масок по порогу
        filtered_masks = []
        for mask_data in self.current_masks:
            if mask_data['conf'] >= confidence:
                filtered_masks.append(mask_data)

        self.current_masks = filtered_masks

        # Обновление таблицы
        self.results_table.update_data(self.current_masks, self.original_image)

        # Обновление визуализаций
        self.create_contour_visualizations()

        self.status_bar.showMessage(f"Применен порог {confidence:.2f}, осталось {len(self.current_masks)} объектов")

    def on_item_selected(self, idx):
        """Обработка выбора элемента из таблицы"""
        if idx is not None and idx < len(self.current_masks):
            self.highlight_mask(self.current_masks[idx])

    def highlight_mask(self, mask_data):
        """Подсветка выбранной маски на изображении результата"""
        if self.original_image is None or self.current_result is None:
            return

        h, w = self.original_image.shape[:2]
        mask = mask_data['mask']

        # Масштабирование маски
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Создание подсвеченного изображения
        highlighted = self.original_image.copy()
        highlighted[mask_binary == 1] = [0, 0, 255]  # Красный цвет для подсветки

        # Отображение в последнем слоте (временно)
        self.display_image_in_label(3, highlighted)

        # Возврат к нормальному виду через 2 секунды
        QTimer.singleShot(2000, lambda: self.display_image_in_label(3,
                                                                    self.current_result.plot() if self.current_result else highlighted))

    def save_full_image(self):
        """Сохранение полного изображения"""
        if self.current_result is None:
            QMessageBox.warning(self, "Ошибка", "Нет результатов для сохранения.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            "result.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )

        if file_path:
            plotted = self.current_result.plot()
            cv2.imwrite(file_path, plotted)
            self.status_bar.showMessage(f"Изображение сохранено: {file_path}")

    def save_csv(self):
        """Сохранение результатов в CSV"""
        if not self.current_masks:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить CSV",
            "results.csv",
            "CSV (*.csv)"
        )

        if file_path:
            import csv
            h, w = self.original_image.shape[:2] if self.original_image is not None else (1, 1)

            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Номер', 'Класс', 'Уверенность', 'Площадь (пикс)', 'Радиус (пикс)',
                                 'Периметр (пикс)', 'Компактность', 'Центр X', 'Центр Y'])

                for i, mask_data in enumerate(self.current_masks):
                    area = self.calculate_mask_area(mask_data['mask'], w, h)
                    radius = np.sqrt(area / np.pi)
                    perimeter = self.results_table.calculate_mask_perimeter(mask_data['mask'], w, h)
                    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                    center_x, center_y = self.results_table.calculate_center(mask_data['mask'], w, h)

                    writer.writerow([
                        i + 1,
                        mask_data['class_name'],
                        f"{mask_data['conf']:.3f}",
                        f"{area:.0f}",
                        f"{radius:.1f}",
                        f"{perimeter:.1f}",
                        f"{compactness:.3f}",
                        f"{center_x:.1f}",
                        f"{center_y:.1f}"
                    ])

            self.status_bar.showMessage(f"CSV сохранен: {file_path}")

    def clear_memory(self):
        """Очистка памяти"""
        self.current_image = None
        self.current_result = None
        self.current_masks = []

        for label in self.image_labels:
            label.clear()

        self.results_table.clear()

        # Очистка GPU памяти
        torch.cuda.empty_cache()
        gc.collect()

    def show_about(self):
        """Показ информации о программе"""
        QMessageBox.about(
            self,
            "О программе",
            """<h3>Spheroid Segmentation with YOLO</h3>
            <p>Версия: 1.0</p>
            <p>Программа для сегментации сфероидов на изображениях с микроскопа.</p>
            <p>Используемые технологии:</p>
            <ul>
                <li>YOLOv8 для сегментации</li>
                <li>PyQt5 для интерфейса</li>
                <li>OpenCV для обработки изображений</li>
            </ul>
            <p>© 2024</p>
            """
        )

    def closeEvent(self, event):
        """Обработка закрытия окна"""
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