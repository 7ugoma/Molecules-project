import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import gc
from ultralytics import YOLO

model = YOLO("best.pt")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Spheroid Segmentation with YOLO")

        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.load_image)
        file_menu.add_command(label="Save Full Image", command=self.save_full_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        root.config(menu=menubar)

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        right_frame = tk.Frame(main_frame, width=250)
        right_frame.pack(side="right", fill="y")

        self.canvas = tk.Canvas(left_frame, width=800, height=800, bg="black")
        self.canvas.pack(fill="both", expand=True)

        controls = tk.Frame(left_frame)
        controls.pack()
        tk.Button(controls, text="Run", command=self.run_model).pack(side="left")

        threshold_frame = tk.Frame(left_frame)
        threshold_frame.pack(fill="x", pady=5)
        tk.Label(threshold_frame, text="Confidence threshold (0-1):").pack(side="left")
        self.threshold_entry = tk.Entry(threshold_frame, width=5)
        self.threshold_entry.insert(0, "0.5")
        self.threshold_entry.pack(side="left", padx=5)
        tk.Button(threshold_frame, text="Apply", command=self.apply_threshold).pack(side="left")

        self.tree = ttk.Treeview(right_frame, columns=("number","class", "conf", "area"), show="headings")
        self.tree.heading("number", text="#")
        self.tree.heading("class", text="Class")
        self.tree.heading("conf", text="Confidence")
        self.tree.heading("area", text="Area")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_table_select)
        self.tree.bind("<Motion>", self.on_table_hover)
        self.tree_last_hover = None

        self.image = None
        self.display_img = None
        self.tk_img = None
        self.base_result = None
        self.masks = []
        self.selected_idx = None
        self.crop_win = None

        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.do_zoom)

    def clear_memory(self):
        self.canvas.delete("all")
        self.tk_img = None
        self.display_img = None
        self.image = None
        self.base_result = None
        self.masks.clear()
        self.selected_idx = None
        self.tree.delete(*self.tree.get_children())
        if self.crop_win:
            self.crop_win.destroy()
            self.crop_win = None
        torch.cuda.empty_cache()
        gc.collect()

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.clear_memory()
        img = cv2.imread(path)
        if img is None:
            return
        self.image = cv2.resize(img, (1024, 1024))
        self.display_img = cv2.resize(self.image, (800, 800))
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.render(self.display_img)

    def run_model(self):
        if self.image is None:
            return
        with torch.no_grad():
            self.base_result = model(self.image)[0]
        self.apply_threshold()

    def apply_threshold(self):
        if self.image is None:
            return
        if self.base_result is None:
            self.display_img = cv2.resize(self.image, (800, 800))
            self.render(self.display_img)
            return
        try:
            threshold = float(self.threshold_entry.get())
            threshold = max(0.0, min(threshold, 1.0))
        except:
            threshold = 0.5

        result = self.base_result
        plotted = result.plot()
        plotted = cv2.resize(plotted, (800,800))
        overlay = plotted.copy()

        self.masks.clear()
        self.tree.delete(*self.tree.get_children())

        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()
            boxes = result.boxes
            classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else np.zeros(len(boxes))

            number = 1
            for i, mask in enumerate(masks_data):
                conf = float(boxes[i].conf)
                if conf < threshold:
                    continue
                class_idx = int(classes[i])
                class_name = result.names[class_idx] if hasattr(result, 'names') else f"Class {class_idx}"

                mask_resized = cv2.resize(mask, (self.display_img.shape[1], self.display_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                ys, xs = np.where(mask_binary == 1)
                area = len(xs)

                self.masks.append((mask_binary, conf, i, area, class_name, number))
                self.tree.insert("", "end", iid=str(i), values=(number, class_name, f"{conf:.2f}", area))

                if len(xs)>0 and len(ys)>0:
                    cx, cy = xs.mean(), ys.mean()
                    cv2.putText(overlay, str(number), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255,0,0), 2)
                number +=1

        self.display_img = overlay
        self.render(self.display_img)

    def on_table_hover(self, event):
        row_id = self.tree.identify_row(event.y)
        if row_id == self.tree_last_hover:
            return
        self.tree_last_hover = row_id

        if row_id and self.display_img is not None:
            idx = int(row_id)
            for mask, conf, i, area, class_name, number in self.masks:
                if i == idx:
                    overlay = self.display_img.copy()
                    overlay[mask == 1] = [0, 255, 0]
                    self.render(overlay)
                    return
        elif self.display_img is not None:
            self.render(self.display_img)

    def on_click(self, event):
        if not self.masks or self.display_img is None:
            return
        x = int((event.x - self.pan_x) / self.zoom)
        y = int((event.y - self.pan_y) / self.zoom)
        for mask, conf, idx, area, class_name, number in self.masks:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] == 1:
                self.selected_idx = idx
                self.highlight_selected(mask)
                self.show_crop(idx, mask, conf, area, class_name, number)
                break

    def highlight_selected(self, mask):
        overlay = self.display_img.copy()
        overlay[mask == 1] = [255, 0, 0]
        self.render(overlay)

    def on_table_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        for mask, conf, i, area, class_name, number in self.masks:
            if i == idx:
                self.highlight_selected(mask)
                self.show_crop(i, mask, conf, area, class_name, number)
                break

    def show_crop(self, idx, mask, conf, area, class_name, number):
        if self.crop_win:
            self.crop_win.destroy()

        mask_for_crop = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked = self.image.copy()
        masked[mask_for_crop == 0] = 0

        ys, xs = np.where(mask_for_crop == 1)
        if len(xs) == 0:
            return
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = masked[y1:y2, x1:x2]

        max_size = 280
        h, w = crop.shape[:2]
        scale = max_size / max(h, w)
        new_w, new_h = int(w*scale), int(h*scale)
        crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        win = tk.Toplevel(self.root)
        self.crop_win = win
        win.title(f"{class_name} {number}")
        win.geometry(f"{max_size}x{max_size+150}")
        win.resizable(False, False)

        def start_move(e):
            win._x, win._y = e.x, e.y
        def move(e):
            dx, dy = e.x - win._x, e.y - win._y
            win.geometry(f"+{win.winfo_x()+dx}+{win.winfo_y()+dy}")
        win.bind("<Button-1>", start_move)
        win.bind("<B1-Motion>", move)

        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(crop_rgb))
        lbl = tk.Label(win, image=img)
        lbl.image = img
        lbl.pack(pady=5)

        info_frame = tk.Frame(win)
        info_frame.pack()
        tk.Label(info_frame, text=f"Class: {class_name}").pack()
        tk.Label(info_frame, text=f"Conf: {conf:.2f}").pack()
        tk.Label(info_frame, text=f"Area: {area}").pack()

        def save_crop():
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                initialfile=f"{class_name}_{number}.png")
            if path:
                cv2.imwrite(path, crop)
        tk.Button(win, text="Save", command=save_crop).pack(pady=5)

    def save_full_image(self):
        if self.display_img is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, self.display_img)

    def render(self, img):
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        w, h = pil.size
        pil = pil.resize((int(w*self.zoom), int(h*self.zoom)))
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_img)

    def do_zoom(self, event):
        if self.display_img is None:
            return
        self.zoom *= 1.1 if event.delta > 0 else 0.9
        self.zoom = max(0.2, min(self.zoom, 10))
        self.render(self.display_img)

    def start_pan(self, event):
        self.drag_start = (event.x, event.y)

    def do_pan(self, event):
        if not self.drag_start or self.display_img is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start = (event.x, event.y)
        self.render(self.display_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()