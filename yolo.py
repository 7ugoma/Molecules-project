import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import gc
from ultralytics import YOLO

# Загрузка предварительно обученной модели YOLO для сегментации
model = YOLO("best.pt")

class App:
    def __init__(self, root):
        """Инициализация главного окна приложения"""
        self.root = root
        self.root.title("Spheroid Segmentation with YOLO")

        # ========== СОЗДАНИЕ МЕНЮ ==========
        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.load_image)      # Открыть изображение
        file_menu.add_command(label="Save Full Image", command=self.save_full_image)  # Сохранить результат
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)            # Выход
        menubar.add_cascade(label="File", menu=file_menu)
        root.config(menu=menubar)

        # ========== ОСНОВНАЯ ПАНЕЛЬ ==========
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # Левая панель - для отображения изображения
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        # Правая панель - для таблицы с результатами (фиксированная ширина 250px)
        right_frame = tk.Frame(main_frame, width=250)
        right_frame.pack(side="right", fill="y")

        # Канвас для отображения изображения с возможностью панорамирования и зума
        self.canvas = tk.Canvas(left_frame, width=800, height=800, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Панель управления
        controls = tk.Frame(left_frame)
        controls.pack()
        tk.Button(controls, text="Run", command=self.run_model).pack(side="left")  # Запуск модели

        # Панель для настройки порога уверенности
        threshold_frame = tk.Frame(left_frame)
        threshold_frame.pack(fill="x", pady=5)
        tk.Label(threshold_frame, text="Confidence threshold (0-1):").pack(side="left")
        self.threshold_entry = tk.Entry(threshold_frame, width=5)
        self.threshold_entry.insert(0, "0.5")  # Значение по умолчанию
        self.threshold_entry.pack(side="left", padx=5)
        tk.Button(threshold_frame, text="Apply", command=self.apply_threshold).pack(side="left")

        # ========== ТАБЛИЦА РЕЗУЛЬТАТОВ ==========
        # Колонки: номер, класс, уверенность, площадь
        self.tree = ttk.Treeview(right_frame, columns=("number","class", "conf", "area"), show="headings")
        self.tree.heading("number", text="#")
        self.tree.heading("class", text="Class")
        self.tree.heading("conf", text="Confidence")
        self.tree.heading("area", text="Area")
        self.tree.pack(fill="both", expand=True)
        
        # Привязка событий таблицы
        self.tree.bind("<<TreeviewSelect>>", self.on_table_select)  # Выбор строки
        self.tree.bind("<Motion>", self.on_table_hover)             # Наведение мыши
        self.tree_last_hover = None  # Для отслеживания последней строки при наведении

        # ========== ПЕРЕМЕННЫЕ ДЛЯ ХРАНЕНИЯ ДАННЫХ ==========
        self.image = None          # Оригинальное изображение (1024x1024)
        self.display_img = None    # Изображение для отображения (800x800)
        self.tk_img = None         # Tkinter-совместимое изображение
        self.base_result = None    # Результат работы модели
        self.masks = []            # Список масок: [(маска, уверенность, индекс, площадь, класс, номер), ...]
        self.selected_idx = None   # Индекс выбранного объекта
        self.crop_win = None       # Окно с обрезанным объектом

        # ========== НАСТРОЙКИ ПАНОРАМИРОВАНИЯ И ЗУМА ==========
        self.zoom = 1.0            # Коэффициент увеличения
        self.pan_x = 0             # Смещение по X
        self.pan_y = 0             # Смещение по Y
        self.drag_start = None     # Начальная точка перетаскивания

        # ========== ПРИВЯЗКА СОБЫТИЙ КАНВАСА ==========
        self.canvas.bind("<Button-1>", self.on_click)           # Клик левой кнопкой - выбор объекта
        self.canvas.bind("<ButtonPress-3>", self.start_pan)     # Зажатие правой кнопки - начало панорамирования
        self.canvas.bind("<B3-Motion>", self.do_pan)            # Движение с зажатой правой - панорамирование
        self.canvas.bind("<MouseWheel>", self.do_zoom)          # Колесико мыши - масштабирование

    def clear_memory(self):
        """Очистка памяти при загрузке нового изображения"""
        self.canvas.delete("all")           # Очистка канваса
        self.tk_img = None                  # Удаление Tkinter-изображения
        self.display_img = None             # Удаление отображаемого изображения
        self.image = None                   # Удаление оригинального изображения
        self.base_result = None             # Удаление результатов модели
        self.masks.clear()                  # Очистка списка масок
        self.selected_idx = None            # Сброс выбранного индекса
        self.tree.delete(*self.tree.get_children())  # Очистка таблицы
        
        # Закрытие окна с обрезанным объектом, если открыто
        if self.crop_win:
            self.crop_win.destroy()
            self.crop_win = None
        
        # Очистка памяти GPU и сборка мусора
        torch.cuda.empty_cache()
        gc.collect()

    def load_image(self):
        """Загрузка изображения через диалоговое окно"""
        path = filedialog.askopenfilename()
        if not path:
            return
        
        self.clear_memory()                 # Очистка памяти перед загрузкой
        
        img = cv2.imread(path)              # Чтение изображения
        if img is None:
            return
        
        # Изменение размера для обработки моделью и отображения
        self.image = cv2.resize(img, (1024, 1024))      # Для модели
        self.display_img = cv2.resize(self.image, (800, 800))  # Для отображения
        
        # Сброс параметров масштабирования
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self.render(self.display_img)       # Отображение изображения

    def run_model(self):
        """Запуск модели YOLO для сегментации"""
        if self.image is None:
            return
        
        # Выполнение инференса без вычисления градиентов (экономия памяти)
        with torch.no_grad():
            self.base_result = model(self.image)[0]
        
        self.apply_threshold()              # Применение порога уверенности

    def apply_threshold(self):
        """
        Применение порога уверенности к результатам модели.
        Отображает только те объекты, чья уверенность выше порога.
        """
        if self.image is None:
            return
        
        # Если модель еще не запущена - просто отображаем изображение
        if self.base_result is None:
            self.display_img = cv2.resize(self.image, (800, 800))
            self.render(self.display_img)
            return
        
        # Получение и валидация значения порога
        try:
            threshold = float(self.threshold_entry.get())
            threshold = max(0.0, min(threshold, 1.0))
        except:
            threshold = 0.5

        result = self.base_result
        plotted = result.plot()                     # Отрисовка результатов на изображении
        plotted = cv2.resize(plotted, (800,800))    # Изменение размера для отображения
        overlay = plotted.copy()                    # Копия для наложения подписей

        # Очистка предыдущих данных
        self.masks.clear()
        self.tree.delete(*self.tree.get_children())

        # Обработка масок сегментации (если есть)
        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()  # Маски объектов
            boxes = result.boxes                          # Bounding boxes
            classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else np.zeros(len(boxes))

            number = 1  # Нумерация объектов (начинается с 1)
            
            for i, mask in enumerate(masks_data):
                conf = float(boxes[i].conf)  # Уверенность объекта
                
                # Пропускаем объекты с уверенностью ниже порога
                if conf < threshold:
                    continue
                
                # Получение имени класса
                class_idx = int(classes[i])
                class_name = result.names[class_idx] if hasattr(result, 'names') else f"Class {class_idx}"

                # Изменение размера маски под размер отображаемого изображения
                mask_resized = cv2.resize(mask, (self.display_img.shape[1], self.display_img.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)  # Бинаризация маски
                
                # Вычисление площади объекта (количество пикселей)
                ys, xs = np.where(mask_binary == 1)
                area = len(xs)

                # Сохранение данных объекта
                self.masks.append((mask_binary, conf, i, area, class_name, number))
                
                # Добавление строки в таблицу
                self.tree.insert("", "end", iid=str(i), values=(number, class_name, f"{conf:.2f}", area))

                # Нанесение номера на изображение (центр объекта)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = xs.mean(), ys.mean()  # Центр масс объекта
                    cv2.putText(overlay, str(number), (int(cx), int(cy)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                
                number += 1  # Увеличение счетчика

        self.display_img = overlay
        self.render(self.display_img)

    def on_table_hover(self, event):
        """
        Обработка наведения мыши на строку таблицы.
        Подсвечивает соответствующий объект на изображении зеленым цветом.
        """
        row_id = self.tree.identify_row(event.y)  # Определение строки под курсором
        
        # Если та же строка - ничего не делаем
        if row_id == self.tree_last_hover:
            return
        
        self.tree_last_hover = row_id

        if row_id and self.display_img is not None:
            idx = int(row_id)  # Индекс объекта в результатах модели
            # Поиск маски по индексу
            for mask, conf, i, area, class_name, number in self.masks:
                if i == idx:
                    overlay = self.display_img.copy()
                    overlay[mask == 1] = [0, 255, 0]  # Заливка маски зеленым цветом
                    self.render(overlay)
                    return
        # Если курсор не над строкой - возвращаем обычное изображение
        elif self.display_img is not None:
            self.render(self.display_img)

    def on_click(self, event):
        """
        Обработка клика по изображению.
        Выбирает объект, на который кликнули, и показывает его увеличенное изображение.
        """
        if not self.masks or self.display_img is None:
            return
        
        # Преобразование координат клика с учетом панорамирования и масштаба
        x = int((event.x - self.pan_x) / self.zoom)
        y = int((event.y - self.pan_y) / self.zoom)
        
        # Поиск объекта, содержащего точку клика
        for mask, conf, idx, area, class_name, number in self.masks:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] == 1:
                self.selected_idx = idx
                self.highlight_selected(mask)  # Подсветка выбранного объекта красным
                self.show_crop(idx, mask, conf, area, class_name, number)  # Показ увеличенного объекта
                break

    def highlight_selected(self, mask):
        """Подсветка выбранного объекта красным цветом"""
        overlay = self.display_img.copy()
        overlay[mask == 1] = [255, 0, 0]  # Заливка маски красным
        self.render(overlay)

    def on_table_select(self, event):
        """Обработка выбора строки в таблице"""
        sel = self.tree.selection()  # Получение выбранных строк
        if not sel:
            return
        
        idx = int(sel[0])  # Индекс выбранного объекта
        
        # Поиск объекта по индексу и его отображение
        for mask, conf, i, area, class_name, number in self.masks:
            if i == idx:
                self.highlight_selected(mask)
                self.show_crop(i, mask, conf, area, class_name, number)
                break

    def show_crop(self, idx, mask, conf, area, class_name, number):
        """
        Создание отдельного окна с увеличенным изображением выбранного объекта.
        
        Параметры:
        - idx: индекс объекта в результатах модели
        - mask: бинарная маска объекта
        - conf: уверенность модели
        - area: площадь объекта в пикселях
        - class_name: имя класса
        - number: порядковый номер объекта
        """
        # Закрываем предыдущее окно, если оно открыто
        if self.crop_win:
            self.crop_win.destroy()

        # Изменение размера маски под оригинальное изображение
        mask_for_crop = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Вырезаем объект, обнуляя все, что не входит в маску
        masked = self.image.copy()
        masked[mask_for_crop == 0] = 0

        # Определение bounding box объекта
        ys, xs = np.where(mask_for_crop == 1)
        if len(xs) == 0:
            return
        
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = masked[y1:y2, x1:x2]  # Обрезанное изображение

        # Изменение размера для отображения (максимум 280px)
        max_size = 280
        h, w = crop.shape[:2]
        scale = max_size / max(h, w)
        new_w, new_h = int(w*scale), int(h*scale)
        crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Создание нового окна
        win = tk.Toplevel(self.root)
        self.crop_win = win
        win.title(f"{class_name} {number}")
        win.geometry(f"{max_size}x{max_size+150}")
        win.resizable(False, False)

        # Функции для перемещения окна
        def start_move(e):
            win._x, win._y = e.x, e.y
        
        def move(e):
            dx, dy = e.x - win._x, e.y - win._y
            win.geometry(f"+{win.winfo_x()+dx}+{win.winfo_y()+dy}")
        
        win.bind("<Button-1>", start_move)
        win.bind("<B1-Motion>", move)

        # Отображение изображения объекта
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(crop_rgb))
        lbl = tk.Label(win, image=img)
        lbl.image = img  # Сохранение ссылки для предотвращения сборки мусора
        lbl.pack(pady=5)

        # Информационная панель
        info_frame = tk.Frame(win)
        info_frame.pack()
        tk.Label(info_frame, text=f"Class: {class_name}").pack()
        tk.Label(info_frame, text=f"Conf: {conf:.2f}").pack()
        tk.Label(info_frame, text=f"Area: {area}").pack()

        # Кнопка сохранения объекта
        def save_crop():
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                initialfile=f"{class_name}_{number}.png")
            if path:
                cv2.imwrite(path, crop)
        
        tk.Button(win, text="Save", command=save_crop).pack(pady=5)

    def save_full_image(self):
        """Сохранение всего изображения с наложенными результатами"""
        if self.display_img is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, self.display_img)

    def render(self, img):
        """
        Отрисовка изображения на канвасе с учетом текущего масштаба и панорамирования.
        
        Параметры:
        - img: изображение в формате BGR (OpenCV)
        """
        if img is None:
            return
        
        # Конвертация из BGR в RGB для PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        
        # Применение масштабирования
        w, h = pil.size
        pil = pil.resize((int(w*self.zoom), int(h*self.zoom)))
        
        # Конвертация в формат Tkinter
        self.tk_img = ImageTk.PhotoImage(pil)
        
        # Отрисовка на канвасе с учетом панорамирования
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_img)

    def do_zoom(self, event):
        """Масштабирование изображения с помощью колесика мыши"""
        if self.display_img is None:
            return
        
        # Изменение коэффициента масштаба
        self.zoom *= 1.1 if event.delta > 0 else 0.9
        self.zoom = max(0.2, min(self.zoom, 10))  # Ограничение от 0.2 до 10
        
        self.render(self.display_img)

    def start_pan(self, event):
        """Начало панорамирования (зажата правая кнопка мыши)"""
        self.drag_start = (event.x, event.y)

    def do_pan(self, event):
        """Панорамирование изображения"""
        if not self.drag_start or self.display_img is None:
            return
        
        # Расчет смещения
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.pan_x += dx
        self.pan_y += dy
        
        # Обновление начальной точки для следующего кадра
        self.drag_start = (event.x, event.y)
        
        self.render(self.display_img)


# ========== ТОЧКА ВХОДА ==========
if __name__ == "__main__":
    root = tk.Tk()      # Создание корневого окна Tkinter
    app = App(root)     # Создание экземпляра приложения
    root.mainloop()     # Запуск главного цикла обработки событий