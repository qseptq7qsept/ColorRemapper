import sys
import os
import json
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageFilter
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QColorDialog,
    QSpinBox,
    QSlider,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def brightness(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def extract_palette(img: Image.Image, n_colors: int) -> List[Tuple[int, int, int]]:
    pal = img.quantize(colors=n_colors, method=Image.FASTOCTREE)
    raw = pal.getpalette()[: n_colors*3]
    cols = [tuple(raw[i:i+3]) for i in range(0, len(raw), 3)]
    seen = set(); uniq = []
    for c in cols:
        if c not in seen:
            seen.add(c); uniq.append(c)
    while len(uniq) < n_colors:
        uniq.append((0,0,0))
    return uniq[:n_colors]

def pil2pixmap(img: Image.Image, max_dim: int = 350) -> QPixmap:
    img_c = img.copy()
    img_c.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return QPixmap.fromImage(ImageQt(img_c))

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))


# --------------------------------------------------------------------
# Draggable List
# --------------------------------------------------------------------

class DraggableList(QListWidget):
    def __init__(self, on_reorder=None, parent=None):
        super().__init__(parent)
        self.on_reorder = on_reorder
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QListWidget.InternalMove)

    def dropEvent(self, event):
        super().dropEvent(event)
        if callable(self.on_reorder):
            self.on_reorder()


# --------------------------------------------------------------------
# Main GUI
# --------------------------------------------------------------------

class RemapperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ColorRemapper v1.0 -q7")
        self.resize(1000,600)

        self.source_img: Optional[Image.Image] = None
        self.target_img: Optional[Image.Image] = None
        self.result_img: Optional[Image.Image] = None

        self.n_clusters = 5
        self.blur_amount = 0.0

        self.source_palette: List[Tuple[int,int,int]] = []
        self.orig_target_palette: List[Tuple[int,int,int]] = []
        self.target_palette: List[Tuple[int,int,int]] = []

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # --- toolbar ---
        btn_load_src = QPushButton("Load Source")
        btn_load_tgt = QPushButton("Load Target")
        btn_load_map = QPushButton("Load JSON")
        self.spin = QSpinBox(); self.spin.setRange(2,32); self.spin.setValue(self.n_clusters)
        self.spin.setPrefix("Colors : ")

        # blur slider now 0–50 representing 0.0–10.0
        self.lbl_blur = QLabel("Blur: 0.0")
        self.slider_blur = QSlider(Qt.Horizontal)
        self.slider_blur.setRange(0, 50)
        self.slider_blur.setValue(0)
        self.slider_blur.setFixedWidth(140)

        btn_json  = QPushButton("Export JSON")
        btn_save  = QPushButton("Save Result")
        btn_batch = QPushButton("Batch Process")

        top = QHBoxLayout()
        for w in (btn_load_src, btn_load_tgt, btn_load_map, self.spin):
            top.addWidget(w)
        top.addWidget(self.lbl_blur)
        top.addWidget(self.slider_blur)
        for w in (btn_json, btn_save, btn_batch):
            top.addWidget(w)
        top.addStretch(1)

        # --- palettes ---
        self.list_source = DraggableList(on_reorder=self.compute_result)
        self.list_target = DraggableList(on_reorder=self.compute_result)

        palettes = QHBoxLayout()
        source_layout = QVBoxLayout()
        lbl_sp = QLabel("Source Palette"); lbl_sp.setAlignment(Qt.AlignCenter)
        source_layout.addWidget(lbl_sp); source_layout.addWidget(self.list_source)
        target_layout = QVBoxLayout()
        lbl_tp = QLabel("Target Palette"); lbl_tp.setAlignment(Qt.AlignCenter)
        target_layout.addWidget(lbl_tp); target_layout.addWidget(self.list_target)
        palettes.addLayout(source_layout)
        palettes.addLayout(target_layout)

        # --- image display ---
        self.lbl_source = QLabel("Source")
        self.lbl_target = QLabel("Target")
        self.lbl_result = QLabel("Output")
        for L in (self.lbl_source, self.lbl_target, self.lbl_result):
            L.setAlignment(Qt.AlignCenter)
            L.setStyleSheet("border:1px solid #444;")
            L.setMinimumSize(200,200)
        imgs = QHBoxLayout()
        imgs.addWidget(self.lbl_source)
        imgs.addWidget(self.lbl_target)
        imgs.addWidget(self.lbl_result)

        # --- assemble ---
        main = QVBoxLayout(central)
        main.addLayout(top)
        main.addLayout(palettes)
        main.addLayout(imgs)

        # --- connections ---
        btn_load_src.clicked.connect(self.load_source)
        btn_load_tgt.clicked.connect(self.load_target)
        btn_load_map.clicked.connect(self.load_json)
        self.spin.valueChanged.connect(self.refresh_palettes)
        self.slider_blur.valueChanged.connect(self.on_blur_changed)
        btn_json.clicked.connect(self.export_json)
        btn_save.clicked.connect(self.save_result)
        btn_batch.clicked.connect(self.batch_process)

        self.list_source.itemDoubleClicked.connect(self.edit_source_color)
        self.list_target.itemDoubleClicked.connect(self.edit_target_color)

    def on_blur_changed(self, v: int):
        # convert slider step to 0.2 increments
        self.blur_amount = v * 0.2
        self.lbl_blur.setText(f"Blur: {self.blur_amount:.1f}")
        self.compute_result()

    def load_source(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Source", "", "Images (*.png *.jpg *.jpeg)")
        if not f: return
        self.source_img = Image.open(f).convert("RGBA")
        self.lbl_source.setPixmap(pil2pixmap(self.source_img))
        self.refresh_palettes()

    def load_target(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Target", "", "Images (*.png *.jpg *.jpeg)")
        if not f: return
        self.target_img = Image.open(f).convert("RGBA")
        self.lbl_target.setPixmap(pil2pixmap(self.target_img))
        self.refresh_palettes()

    def load_json(self):
        f, _ = QFileDialog.getOpenFileName(self, "Import JSON", "", "JSON (*.json)")
        if not f: return
        try:
            data = json.load(open(f, encoding='utf-8'))
            mapping_hex = data.get("mapping", data)
            self.blur_amount = float(data.get("blur", 0.0))
        except Exception:
            QMessageBox.warning(self, "Invalid JSON", "Failed to load JSON.")
            return

        # update slider & label from JSON
        self.slider_blur.setValue(int(self.blur_amount / 0.2))
        self.lbl_blur.setText(f"Blur: {self.blur_amount:.1f}")

        self.source_palette.clear()
        self.target_palette.clear()
        for t_hex, s_hex in mapping_hex.items():
            self.target_palette.append(hex_to_rgb(t_hex))
            self.source_palette.append(hex_to_rgb(s_hex))

        self.populate_lists()
        self.compute_result()
        QMessageBox.information(self, "JSON Loaded",
                                f"Imported {len(self.target_palette)} mappings with blur={self.blur_amount:.1f}")

    def refresh_palettes(self):
        if not (self.source_img and self.target_img):
            return
        self.n_clusters = self.spin.value()
        self.source_palette      = sorted(extract_palette(self.source_img, self.n_clusters), key=brightness)
        self.orig_target_palette = sorted(extract_palette(self.target_img, self.n_clusters), key=brightness)
        self.target_palette      = list(self.orig_target_palette)
        self.populate_lists()
        self.compute_result()

    def populate_lists(self):
        self.list_source.clear()
        self.list_target.clear()
        for col in self.source_palette:
            it = QListWidgetItem(rgb_to_hex(col))
            it.setBackground(QColor(*col))
            it.setForeground(QColor(*(255-c for c in col)))
            it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|Qt.ItemIsDragEnabled)
            it.setData(Qt.UserRole, col)
            self.list_source.addItem(it)
        for col in self.target_palette:
            it = QListWidgetItem(rgb_to_hex(col))
            it.setBackground(QColor(*col))
            it.setForeground(QColor(*(255-c for c in col)))
            it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|Qt.ItemIsDragEnabled)
            it.setData(Qt.UserRole, col)
            self.list_target.addItem(it)

    def sync_palettes(self):
        self.source_palette = [self.list_source.item(i).data(Qt.UserRole) for i in range(self.list_source.count())]
        self.target_palette = [self.list_target.item(i).data(Qt.UserRole) for i in range(self.list_target.count())]

    def edit_source_color(self, item: QListWidgetItem):
        orig = item.data(Qt.UserRole)
        col = QColorDialog.getColor(QColor(*orig), self)
        if not col.isValid(): return
        rgb = (col.red(),col.green(),col.blue())
        item.setBackground(col); item.setText(rgb_to_hex(rgb))
        item.setForeground(QColor(*(255-c for c in rgb)))
        item.setData(Qt.UserRole, rgb)
        self.compute_result()

    def edit_target_color(self, item: QListWidgetItem):
        orig = item.data(Qt.UserRole)
        col = QColorDialog.getColor(QColor(*orig), self)
        if not col.isValid(): return
        rgb = (col.red(),col.green(),col.blue())
        item.setBackground(col); item.setText(rgb_to_hex(rgb))
        item.setForeground(QColor(*(255-c for c in rgb)))
        item.setData(Qt.UserRole, rgb)
        self.compute_result()

    def apply_mapping_to_image(
        self, img: Image.Image,
        mapping: Dict[Tuple[int,int,int], Tuple[int,int,int]]
    ) -> Image.Image:
        src = img.convert('RGBA'); w,h = src.size
        out = Image.new('RGBA',(w,h))
        p_src, p_out = src.load(), out.load()
        for x in range(w):
            for y in range(h):
                r,g,b,a = p_src[x,y]
                if a==0:
                    p_out[x,y] = (r,g,b,a)
                else:
                    if (r,g,b) in mapping:
                        nr,ng,nb = mapping[(r,g,b)]
                    else:
                        closest = min(mapping.keys(),
                                      key=lambda c: (r-c[0])**2 + (g-c[1])**2 + (b-c[2])**2)
                        nr,ng,nb = mapping[closest]
                    p_out[x,y] = (nr,ng,nb,a)
        return out

    def compute_result(self):
        if not (self.target_img and self.source_palette and self.target_palette):
            return
        self.sync_palettes()
        mapping = {self.target_palette[i]: self.source_palette[i]
                   for i in range(len(self.target_palette))}

        remapped = self.apply_mapping_to_image(self.target_img, mapping)

        if self.blur_amount > 0:
            r,g,b,a = remapped.split()
            rgb = Image.merge("RGB",(r,g,b))
            brgb = rgb.filter(ImageFilter.GaussianBlur(radius=self.blur_amount))
            r2,g2,b2 = brgb.split()
            final = Image.merge("RGBA",(r2,g2,b2,a))
        else:
            final = remapped

        self.result_img = final
        self.lbl_result.setPixmap(pil2pixmap(final))

    def save_result(self):
        if not self.result_img:
            QMessageBox.information(self, "Nothing to save", "Load and remap images first.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Result", "result.png",
                                           "PNG (*.png);;JPEG (*.jpg)")
        if not p: return
        fmt = "PNG" if p.lower().endswith(".png") else "JPEG"
        self.result_img.save(p, fmt)

    def export_json(self):
        if not (self.target_palette and self.source_palette):
            QMessageBox.information(self, "Nothing to export", "Load and remap first.")
            return

        self.sync_palettes()
        mapping_hex = {rgb_to_hex(t): rgb_to_hex(s)
                       for t,s in zip(self.target_palette, self.source_palette)}
        payload = {"blur": self.blur_amount, "mapping": mapping_hex}

        p, _ = QFileDialog.getSaveFileName(self, "Export JSON", "map.json", "JSON (*.json)")
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4)

    def batch_process(self):
        if not (self.source_img and self.target_img):
            QMessageBox.information(self, "Batch Failed", "Please load both images first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if not folder: return

        out_dir = os.path.join(folder, "Remapped")
        os.makedirs(out_dir, exist_ok=True)

        self.sync_palettes()
        mapping_hex = {rgb_to_hex(t): rgb_to_hex(s)
                       for t,s in zip(self.target_palette, self.source_palette)}
        payload = {"blur": self.blur_amount, "mapping": mapping_hex}
        with open(os.path.join(out_dir, "mapping.json"), 'w', encoding='utf-8') as mf:
            json.dump(payload, mf, indent=4)

        count = 0
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png",".jpg",".jpeg")): continue
            try:
                img = Image.open(os.path.join(folder, fname)).convert('RGBA')
            except:
                continue

            remapped = self.apply_mapping_to_image(img, {
                hex_to_rgb(t): hex_to_rgb(s) for t,s in mapping_hex.items()
            })

            if self.blur_amount > 0:
                r,g,b,a = remapped.split()
                rgb = Image.merge("RGB",(r,g,b))
                brgb = rgb.filter(ImageFilter.GaussianBlur(radius=self.blur_amount))
                r2,g2,b2 = brgb.split()
                final = Image.merge("RGBA",(r2,g2,b2,a))
            else:
                final = remapped

            name, ext = os.path.splitext(fname)
            final.save(os.path.join(out_dir, f"{name}_remapped{ext}"))
            count += 1

        QMessageBox.information(self, "Batch Completed",
                                f"Processed {count} images.\nOutputs in: {out_dir}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dark = QPalette()
    dark.setColor(QPalette.Window,          QColor(53,53,53))
    dark.setColor(QPalette.WindowText,      QColor(255,255,255))
    dark.setColor(QPalette.Base,            QColor(42,42,42))
    dark.setColor(QPalette.AlternateBase,   QColor(66,66,66))
    dark.setColor(QPalette.ToolTipBase,     QColor(255,255,255))
    dark.setColor(QPalette.ToolTipText,     QColor(255,255,255))
    dark.setColor(QPalette.Text,            QColor(255,255,255))
    dark.setColor(QPalette.Button,          QColor(53,53,53))
    dark.setColor(QPalette.ButtonText,      QColor(255,255,255))
    dark.setColor(QPalette.BrightText,      QColor(255,0,0))
    dark.setColor(QPalette.Link,            QColor(208,42,218))
    dark.setColor(QPalette.Highlight,       QColor(208,42,218))
    dark.setColor(QPalette.HighlightedText, QColor(0,0,0))
    app.setPalette(dark)

    win = RemapperGUI()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
