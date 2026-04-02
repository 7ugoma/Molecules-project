import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


# --- КЛАСС ДЛЯ ПЕРЕТАСКИВАНИЯ (Drag & Drop) ---
class DropZone(QLabel):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)  # Разрешаем принимать объекты
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed #bbb; background: #f9f9f9; border-radius: 5px;")
        self.parent_app = parent  # Ссылка на главное окно для вызова обработки

    def dragEnterEvent(self, event):
        # Проверяем, что перетаскивают именно файлы
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("border: 2px solid #0277bd; background: #e1f5fe; border-radius: 5px;")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        # Возвращаем стиль, если файл унесли обратно
        self.setStyleSheet("border: 2px dashed #bbb; background: #f9f9f9; border-radius: 5px;")

    def dropEvent(self, event):
        # Получаем путь к файлу при «отпускании» мыши
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            # Проверяем расширение
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.parent_app.start_processing(file_path)
            else:
                QMessageBox.warning(self, "Ошибка", "Пожалуйста, перетащите изображение (jpg, png).")
        self.setStyleSheet("border: 1px solid #999; background: white;")


# --- ЛОГИКА ОБРАБОТКИ ---
def process_image(file_path):
    stats = {"Количество контуров разного размера": "значение из файла", "Процент их насыщенности": "из другого файла",
             "Радиус": "из другого файла", "че-то еще": "что-то"}
    info = {
        "значение": "пояснение"
    }
    return [file_path] * 4, stats, info


# --- ГЛАВНОЕ ОКНО ---
class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Processor 2x2 + Drag&Drop")
        self.resize(1250, 850)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. ЛЕВАЯ ПАНЕЛЬ
        self.stats_panel = QVBoxLayout()
        self.stats_label = QLabel("Загрузите фото кнопкой\nили перетащите в поле 'Слот 1'")
        self.stats_label.setFixedWidth(300)
        self.stats_label.setAlignment(Qt.AlignTop)
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("padding: 15px; background: #fff; border: 1px solid #ccc; border-radius: 8px;")

        self.btn_info = QPushButton("❓ Подробная сводка")
        self.btn_info.setFixedWidth(300)
        self.btn_info.setFixedHeight(40)
        self.btn_info.setEnabled(False)
        self.btn_info.clicked.connect(self.show_summary)

        self.stats_panel.addWidget(self.stats_label)
        self.stats_panel.addWidget(self.btn_info)
        self.stats_panel.addStretch()
        self.main_layout.addLayout(self.stats_panel)

        # 2. ПРАВАЯ ЧАСТЬ
        self.right_container = QVBoxLayout()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(15)

        self.image_widgets = []
        self.caption_widgets = []
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for i in range(4):
            v_box = QVBoxLayout()

            # Первый слот делаем зоной для Drag & Drop
            if i == 0:
                img_label = DropZone("Перетащите сюда оригинал", self)
            else:
                img_label = QLabel(f"Слот {i + 1}")
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setStyleSheet("border: 2px dashed #bbb; background: #f9f9f9; border-radius: 5px;")

            cap_label = QLabel("")
            cap_label.setAlignment(Qt.AlignCenter)
            cap_label.setStyleSheet("font-weight: bold; color: #444; margin-top: 5px;")

            v_box.addWidget(img_label)
            v_box.addWidget(cap_label)

            row, col = positions[i]
            self.grid_layout.addLayout(v_box, row, col)
            self.image_widgets.append(img_label)
            self.caption_widgets.append(cap_label)

        self.btn_upload = QPushButton("📥 Выбрать файл через проводник")
        self.btn_upload.setFixedHeight(50)
        self.btn_upload.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #e1f5fe;")
        self.btn_upload.clicked.connect(self.open_file_dialog)

        self.right_container.addLayout(self.grid_layout)
        self.right_container.addWidget(self.btn_upload)
        self.main_layout.addLayout(self.right_container)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.start_processing(file_path)

    def start_processing(self, file_path):
        # Общая функция для обоих способов загрузки
        paths, stats, info = process_image(file_path)

        titles = [
            "ОРИГИНАЛ",
            "ВСЕ КРУГЛЫЕ КОНТУРЫ",
            "ПОХОЖИЕ КРУГЛЫЕ КОНТУРЫ",
            "ПРОНУМИРОВАННЫЕ СФЕРОИДЫ"
        ]

        self.current_info = info

        for i in range(4):
            pix = QPixmap(paths[i]).scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_widgets[i].setPixmap(pix)
            self.image_widgets[i].setStyleSheet("border: 1px solid #999; background: white;")
            self.caption_widgets[i].setText(titles[i])

        res_html = "<h3>📊 Показатели:</h3><hr>"
        for k, v in stats.items():
            res_html += f"<p><b>{k}:</b> <span style='color: #0277bd;'>{v}</span></p>"
        self.stats_label.setText(res_html)
        self.btn_info.setEnabled(True)

    def show_summary(self):
        summary = "<h3>Справочная информация</h3><br>"
        for k, v in self.current_info.items():
            summary += f"<b>{k}</b>:<br>{v}<br><br>"
        QMessageBox.information(self, "Инфо", summary)


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None: app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    app.exec_()
