"""
Interfaz minima: elegir una o varias imagenes, indicar para cada una el ancho y alto (mm) que abarca
la imagen completa, elegir la operacion (calcular area / medir anchura) y ver el resultado.
Cada operacion vive en su propio modulo: area -> contour_imgpr.py, anchura -> measure_width.py.

Uso:
    python contour_gui.py

En la tabla: doble clic en "Ancho (mm)" o "Alto (mm)" para editar; doble clic en el resto de la fila
(tras procesar) para ver el resultado.
"""

import base64
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

import contour_imgpr as dc
import measure_width as mw
import scale

PREVIEW_MAX = (1000, 700)   # tamano maximo (ancho, alto) de la vista previa en pantalla
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
AUTO = "auto (segun carpeta)"
COLUMNS = ("name", "width", "height", "result")
EDITABLE = {"#2": "width", "#3": "height"}


def op_area(path, profile, size_mm):
    """Calcular area (contour_imgpr). Devuelve (texto_resultado, imagen_resultado | None)."""
    contour, area_px, result, out = dc.process(path, profile, image_size_mm=size_mm)
    if contour is None:
        return "sin contorno", None
    return f"{area_px * scale.pixel_area_mm2(result.shape, size_mm):,.3f} mm2", out


def op_width(path, profile, size_mm):
    """Medir anchura (measure_width). Devuelve (texto_resultado, imagen_resultado | None)."""
    widths, out = mw.measure_width(path, profile, size_mm)
    text = " / ".join(f"{widths[k]:.3f}" for k in ("superior", "curva", "punta")) + " mm"
    return text, out


# nombre en el desplegable -> (funcion, cabecera de la columna de resultado)
OPERATIONS = {
    "Calcular area": (op_area, "Area (mm2)"),
    "Medir anchura": (op_width, "Anchura sup / curva / punta"),
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MSPe-QA - Area del cordon")
        self.minsize(900, 380)
        dc.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # Escritorio/analysis_results
        self.outputs = {}       # iid (ruta completa) -> ruta de la imagen resultado
        self._editor = None

        self.profile_var = tk.StringVar(value=AUTO)
        self.operation_var = tk.StringVar(value=next(iter(OPERATIONS)))
        self.status_var = tk.StringVar(value="Anade imagenes y rellena ancho y alto (mm) de cada una.")

        top = ttk.Frame(self, padding=(10, 10, 10, 0))
        top.pack(fill="x")
        ttk.Button(top, text="Anadir carpeta...", command=self.add_folder).pack(side="left")
        ttk.Button(top, text="Anadir imagenes...", command=self.add_images).pack(side="left", padx=5)
        ttk.Button(top, text="Quitar seleccion", command=self.remove_selected).pack(side="left")
        ttk.Label(top, text="Tipo de imagen:").pack(side="left", padx=(20, 5))
        ttk.Combobox(top, textvariable=self.profile_var, state="readonly", width=24,
                     values=[AUTO, *dc.PROFILES]).pack(side="left")
        ttk.Label(top, text="Operacion:").pack(side="left", padx=(20, 5))
        operation_box = ttk.Combobox(top, textvariable=self.operation_var, state="readonly", width=18,
                                     values=list(OPERATIONS))
        operation_box.pack(side="left")
        operation_box.bind("<<ComboboxSelected>>", self.on_operation_change)
        self.run_btn = ttk.Button(top, text="Procesar", command=self.run)
        self.run_btn.pack(side="right")
        ttk.Button(top, text="Abrir resultados", command=self.open_results).pack(side="right", padx=5)

        mid = ttk.Frame(self, padding=10)
        mid.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(mid, columns=COLUMNS, show="headings", selectmode="extended")
        for col, text, width, anchor in (("name", "Imagen", 280, "w"), ("width", "Ancho (mm)", 90, "center"),
                                         ("height", "Alto (mm)", 90, "center"),
                                         ("result", OPERATIONS[self.operation_var.get()][1], 200, "e")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "name"))
        scroll = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self.on_double_click)

        ttk.Label(self, textvariable=self.status_var, padding=(10, 0, 10, 10)).pack(fill="x")

    # ------------------------------------------------------------------ tabla
    def on_operation_change(self, _event=None):
        self.outputs.clear()
        self.tree.heading("result", text=OPERATIONS[self.operation_var.get()][1])
        for iid in self.tree.get_children():
            self.tree.set(iid, "result", "")

    def _initial_dir(self):
        return dc.FOTOS_DIR if dc.FOTOS_DIR.exists() else Path.home()

    def _insert_images(self, paths):
        for p in paths:
            p = str(Path(p))
            if not self.tree.exists(p):
                self.tree.insert("", "end", iid=p, values=(Path(p).name, "", "", ""))
        if paths:
            self.status_var.set(f"{len(self.tree.get_children())} imagen(es). Rellena ancho y alto (mm).")

    def add_images(self):
        paths = filedialog.askopenfilenames(
            title="Seleccionar imagenes", initialdir=self._initial_dir(),
            filetypes=[("Imagenes", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("Todos", "*.*")])
        self._insert_images(paths)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta de imagenes", initialdir=self._initial_dir())
        if not folder:
            return
        paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            messagebox.showinfo("Sin imagenes", f"No hay imagenes en:\n{folder}")
            return
        self._insert_images(paths)

    def open_results(self):
        dc.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(dc.OUTPUT_DIR)

    def remove_selected(self):
        self._close_editor()
        for iid in self.tree.selection():
            self.outputs.pop(iid, None)
            self.tree.delete(iid)

    def on_double_click(self, event):
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        iid, col = self.tree.identify_row(event.y), self.tree.identify_column(event.x)
        if not iid:
            return
        if col in EDITABLE:
            self._edit_cell(iid, col)
        elif iid in self.outputs:
            self.show_result(self.outputs[iid])

    def _edit_cell(self, iid, col):
        self._close_editor()
        x, y, w, h = self.tree.bbox(iid, col)
        entry = ttk.Entry(self.tree, justify="center")
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, self.tree.set(iid, EDITABLE[col]))
        entry.select_range(0, "end")
        entry.focus_set()

        def commit(_=None):
            if self._editor is entry:
                self.tree.set(iid, EDITABLE[col], entry.get().strip())
                self._close_editor()

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", lambda _: self._close_editor())
        self._editor = entry

    def _close_editor(self):
        if self._editor is not None:
            editor, self._editor = self._editor, None
            editor.destroy()

    # ------------------------------------------------------------ procesado
    def _read_sizes(self):
        """Devuelve {iid: (ancho_mm, alto_mm)} validando todas las filas."""
        sizes = {}
        for iid in self.tree.get_children():
            name = self.tree.set(iid, "name")
            try:
                w = float(self.tree.set(iid, "width").replace(",", "."))
                h = float(self.tree.set(iid, "height").replace(",", "."))
            except ValueError:
                raise ValueError(f"{name}: ancho y alto deben ser numeros (mm).")
            if w <= 0 or h <= 0:
                raise ValueError(f"{name}: ancho y alto deben ser mayores que 0.")
            sizes[iid] = (w, h)
        return sizes

    def run(self):
        self._close_editor()
        if not self.tree.get_children():
            messagebox.showwarning("Falta imagen", "Anade al menos una imagen.")
            return
        try:
            sizes = self._read_sizes()
        except ValueError as e:
            messagebox.showwarning("Datos no validos", str(e))
            return

        profile = None if self.profile_var.get() == AUTO else dc.PROFILES[self.profile_var.get()]
        self.outputs.clear()
        for iid in sizes:
            self.tree.set(iid, "result", "")
        self.run_btn.state(["disabled"])
        operation = OPERATIONS[self.operation_var.get()][0]
        threading.Thread(target=self._worker, args=(sizes, profile, operation), daemon=True).start()

    def _worker(self, sizes, profile, operation):
        total = len(sizes)
        for i, (iid, size_mm) in enumerate(sizes.items(), 1):
            self.after(0, self.status_var.set, f"Procesando {i}/{total}: {Path(iid).name}...")
            try:
                text, out = operation(iid, profile, size_mm)
                self.after(0, self._row_done, iid, text, out)
            except NotImplementedError:
                self.after(0, self._row_set, iid, "pendiente de implementar")
            except Exception as e:  # el error de una imagen no detiene las demas
                self.after(0, self._row_set, iid, f"error: {str(e)[:40]}")
        self.after(0, self._finished)

    def _row_set(self, iid, text):
        if self.tree.exists(iid):
            self.tree.set(iid, "result", text)

    def _row_done(self, iid, text, out):
        self._row_set(iid, text)
        if out is not None and self.tree.exists(iid):
            self.outputs[iid] = out

    def _finished(self):
        self.run_btn.state(["!disabled"])
        self.status_var.set(f"Listo. Resultados en {dc.OUTPUT_DIR}. Doble clic en una fila para ver el contorno.")

    # ------------------------------------------------------------ resultado
    def show_result(self, path):
        img = cv2.imread(str(path))
        if img is None:
            messagebox.showerror("Error", f"No se pudo abrir {path}")
            return
        scale = min(PREVIEW_MAX[0] / img.shape[1], PREVIEW_MAX[1] / img.shape[0], 1.0)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, png = cv2.imencode(".png", img)
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))

        win = tk.Toplevel(self)
        win.title(Path(path).name)
        win.resizable(False, False)
        tk.Label(win, image=photo).pack()
        win._photo = photo  # evita que se libere la imagen


if __name__ == "__main__":
    App().mainloop()
