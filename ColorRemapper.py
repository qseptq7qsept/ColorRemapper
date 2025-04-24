import sys
import os
import json
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageFilter
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QColorDialog,
    QSpinBox, QSlider, QCheckBox, QHBoxLayout, QVBoxLayout,
    QMessageBox,
)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def brightness(rgb: Tuple[int,int,int]) -> float:
    r, g, b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b

def extract_palette(img: Image.Image, n_colors: int) -> List[Tuple[int,int,int]]:
    pal = img.quantize(colors=n_colors, method=Image.FASTOCTREE)
    raw = pal.getpalette()[:n_colors*3]
    cols = [tuple(raw[i:i+3]) for i in range(0, len(raw), 3)]
    seen, uniq = set(), []
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

def rgb_to_hex(rgb: Tuple[int,int,int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

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
        self.setWindowTitle("ColorRemapper v1.1 -q7")
        self.resize(1000,600)

        self.source_img: Optional[Image.Image] = None
        self.target_img: Optional[Image.Image] = None
        self.result_img: Optional[Image.Image] = None

        self.n_clusters  = 5
        self.blur_amount = 0.0

        self.source_palette:      List[Tuple[int,int,int]] = []
        self.orig_target_palette: List[Tuple[int,int,int]] = []
        self.target_palette:      List[Tuple[int,int,int]] = []

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # --- toolbar ---
        btn_load_src = QPushButton("Load Source / Import JSON")
        btn_load_tgt = QPushButton("Load Target")
        btn_load_map = QPushButton("Load JSON (mapping only)")
        self.spin = QSpinBox(); self.spin.setRange(2,32); self.spin.setValue(self.n_clusters)
        self.spin.setPrefix("Colors : ")

        self.lbl_blur    = QLabel("Blur: 0.0")
        self.slider_blur = QSlider(Qt.Horizontal)
        self.slider_blur.setRange(0,50)   # 0→10.0 in steps of 0.2
        self.slider_blur.setValue(0)
        self.slider_blur.setFixedWidth(140)

        self.chk_dither     = QCheckBox("Dither")
        self.chk_antialias  = QCheckBox("Antialias")

        btn_json  = QPushButton("Export JSON")
        btn_save  = QPushButton("Save Result")
        btn_batch = QPushButton("Batch Process")

        top = QHBoxLayout()
        for w in (btn_load_src, btn_load_tgt, btn_load_map, self.spin):
            top.addWidget(w)
        top.addWidget(self.lbl_blur)
        top.addWidget(self.slider_blur)
        top.addWidget(self.chk_dither)
        top.addWidget(self.chk_antialias)
        for w in (btn_json, btn_save, btn_batch):
            top.addWidget(w)
        top.addStretch(1)

        # --- palettes lists ---
        self.list_source = DraggableList(on_reorder=self.compute_result)
        self.list_target = DraggableList(on_reorder=self.compute_result)

        palettes = QHBoxLayout()
        src_layout = QVBoxLayout()
        lbl_sp = QLabel("Source Palette"); lbl_sp.setAlignment(Qt.AlignCenter)
        src_layout.addWidget(lbl_sp); src_layout.addWidget(self.list_source)
        tgt_layout = QVBoxLayout()
        lbl_tp = QLabel("Target Palette"); lbl_tp.setAlignment(Qt.AlignCenter)
        tgt_layout.addWidget(lbl_tp); tgt_layout.addWidget(self.list_target)
        palettes.addLayout(src_layout)
        palettes.addLayout(tgt_layout)

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
        self.slider_blur.valueChanged.connect(lambda v: self.on_blur_changed(v))
        self.chk_dither.stateChanged.connect(self.compute_result)
        self.chk_antialias.stateChanged.connect(self.compute_result)
        btn_json.clicked.connect(self.export_json)
        btn_save.clicked.connect(self.save_result)
        btn_batch.clicked.connect(self.batch_process)

        self.list_source.itemDoubleClicked.connect(self.edit_source_color)
        self.list_target.itemDoubleClicked.connect(self.edit_target_color)

    def on_blur_changed(self, v: int):
        self.blur_amount = v * 0.2
        self.lbl_blur.setText(f"Blur: {self.blur_amount:.1f}")
        self.compute_result()

    def load_source(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Image or Import JSON",
            "",
            "Images (*.png *.jpg *.jpeg);;JSON Settings (*.json)"
        )
        if not f:
            return
        if f.lower().endswith(".json"):
            return self._import_settings(f)

        self.source_img = Image.open(f).convert("RGBA")
        self.lbl_source.setPixmap(pil2pixmap(self.source_img))
        self.refresh_palettes()

    def _import_settings(self, path: str):
        try:
            data = json.load(open(path, encoding='utf-8'))
        except Exception:
            QMessageBox.warning(self, "Invalid JSON", "Failed to load settings JSON.")
            return

        mapping_hex = data.get("mapping", {})
        blur_val    = float(data.get("blur", 0.0))
        clusters    = data.get("clusters", None)
        dither_flag = bool(data.get("dither", False))
        aa_flag     = bool(data.get("antialias", False))

        # apply blur & modes
        self.slider_blur.setValue(int(blur_val/0.2))
        self.on_blur_changed(self.slider_blur.value())
        self.chk_dither.setChecked(dither_flag)
        self.chk_antialias.setChecked(aa_flag)

        if isinstance(clusters, int) and 2 <= clusters <= 32:
            self.spin.setValue(clusters)

        self.source_palette.clear()
        self.target_palette.clear()
        for t_hex, s_hex in mapping_hex.items():
            self.target_palette.append(hex_to_rgb(t_hex))
            self.source_palette.append(hex_to_rgb(s_hex))

        self.populate_lists()
        self.compute_result()
        QMessageBox.information(
            self, "Settings Imported",
            f"Loaded {len(self.target_palette)} pairs, blur={blur_val:.1f}, "
            f"dither={dither_flag}, antialias={aa_flag}"
            + (f", clusters={clusters}" if clusters else "")
        )

    def load_target(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Target", "", "Images (*.png *.jpg *.jpeg)")
        if not f:
            return
        self.target_img = Image.open(f).convert("RGBA")
        self.lbl_target.setPixmap(pil2pixmap(self.target_img))
        self.refresh_palettes()

    def load_json(self):
        f, _ = QFileDialog.getOpenFileName(self, "Import JSON", "", "JSON (*.json)")
        if not f:
            return
        self._import_settings(f)

    def refresh_palettes(self):
        if not (self.source_img and self.target_img):
            return
        self.n_clusters         = self.spin.value()
        self.source_palette     = sorted(extract_palette(self.source_img, self.n_clusters), key=brightness)
        self.orig_target_palette= sorted(extract_palette(self.target_img, self.n_clusters), key=brightness)
        self.target_palette     = list(self.orig_target_palette)
        self.populate_lists()
        self.compute_result()

    def populate_lists(self):
        self.list_source.clear()
        self.list_target.clear()
        for col in self.source_palette:
            it = QListWidgetItem(rgb_to_hex(col))
            it.setBackground(QColor(*col))
            it.setForeground(QColor(*(255-c for c in col)))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
            it.setData(Qt.UserRole, col)
            self.list_source.addItem(it)
        for col in self.target_palette:
            it = QListWidgetItem(rgb_to_hex(col))
            it.setBackground(QColor(*col))
            it.setForeground(QColor(*(255-c for c in col)))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
            it.setData(Qt.UserRole, col)
            self.list_target.addItem(it)

    def sync_palettes(self):
        self.source_palette = [self.list_source.item(i).data(Qt.UserRole)
                               for i in range(self.list_source.count())]
        self.target_palette = [self.list_target.item(i).data(Qt.UserRole)
                               for i in range(self.list_target.count())]

    def edit_source_color(self, item: QListWidgetItem):
        orig = item.data(Qt.UserRole)
        col = QColorDialog.getColor(QColor(*orig), self)
        if not col.isValid():
            return
        rgb = (col.red(), col.green(), col.blue())
        item.setBackground(col)
        item.setText(rgb_to_hex(rgb))
        item.setForeground(QColor(*(255-c for c in rgb)))
        item.setData(Qt.UserRole, rgb)
        self.compute_result()

    def edit_target_color(self, item: QListWidgetItem):
        orig = item.data(Qt.UserRole)
        col = QColorDialog.getColor(QColor(*orig), self)
        if not col.isValid():
            return
        rgb = (col.red(), col.green(), col.blue())
        item.setBackground(col)
        item.setText(rgb_to_hex(rgb))
        item.setForeground(QColor(*(255-c for c in rgb)))
        item.setData(Qt.UserRole, rgb)
        self.compute_result()

    def apply_palette_mapping(
        self,
        img: Image.Image,
        from_palette: List[Tuple[int,int,int]],
        to_palette:   List[Tuple[int,int,int]],
        dither: bool
    ) -> Image.Image:
        """
        Fast palette-quantize + palette-swap:
        1) Strip alpha, quantize RGB → P-mode with dither on/off
        2) Overwrite palette entries with to_palette
        3) Convert back to RGBA, re-attach alpha
        """
        # preserve alpha
        if img.mode == 'RGBA':
            r,g,b,a = img.split()
            rgb   = Image.merge('RGB',(r,g,b))
        else:
            rgb = img.convert('RGB')
            a   = None

        # build palette image
        pal_img = Image.new('P', (1,1))
        flat_from = []
        for c in from_palette:
            flat_from.extend(c)
        flat_from += [0]*(768 - len(flat_from))
        pal_img.putpalette(flat_from)

        # quantize
        d_flag = Image.FLOYDSTEINBERG if dither else Image.NONE
        quant = rgb.quantize(palette=pal_img, dither=d_flag)

        # swap in new palette
        flat_to = []
        for c in to_palette:
            flat_to.extend(c)
        flat_to += [0]*(768 - len(flat_to))
        quant.putpalette(flat_to)

        # back to RGBA
        mapped = quant.convert('RGB')
        if a is not None:
            r2,g2,b2 = mapped.split()
            return Image.merge('RGBA',(r2,g2,b2,a))
        else:
            return mapped.convert('RGBA')

    def compute_result(self):
        if not (self.target_img and self.source_palette and self.target_palette):
            return
        self.sync_palettes()
        from_pal = self.target_palette
        to_pal   = self.source_palette
        do_dither    = self.chk_dither.isChecked()
        do_antialias = self.chk_antialias.isChecked()

        if do_antialias:
            w, h = self.target_img.size
            high = self.target_img.resize((w*2, h*2), resample=Image.NEAREST)
            mapped_high = self.apply_palette_mapping(high, from_pal, to_pal, dither=False)
            final = mapped_high.resize((w, h), resample=Image.LANCZOS)

        elif do_dither:
            final = self.apply_palette_mapping(self.target_img, from_pal, to_pal, dither=True)

        else:
            remapped = self.apply_palette_mapping(self.target_img, from_pal, to_pal, dither=False)
            if self.blur_amount > 0:
                r,g,b,a = remapped.split()
                rgb     = Image.merge('RGB',(r,g,b))
                brgb    = rgb.filter(ImageFilter.GaussianBlur(radius=self.blur_amount))
                r2,g2,b2 = brgb.split()
                final = Image.merge('RGBA',(r2,g2,b2,a))
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
        if not p:
            return
        fmt = "PNG" if p.lower().endswith(".png") else "JPEG"
        self.result_img.save(p, fmt)

    def export_json(self):
        if not (self.target_palette and self.source_palette):
            QMessageBox.information(self, "Nothing to export", "Load and remap first.")
            return
        self.sync_palettes()
        mapping_hex = {
            rgb_to_hex(t): rgb_to_hex(s)
            for t,s in zip(self.target_palette, self.source_palette)
        }
        payload = {
            "clusters":   self.spin.value(),
            "blur":       self.blur_amount,
            "dither":     self.chk_dither.isChecked(),
            "antialias":  self.chk_antialias.isChecked(),
            "mapping":    mapping_hex
        }
        p, _ = QFileDialog.getSaveFileName(self, "Export JSON", "map.json", "JSON (*.json)")
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4)

    def batch_process(self):
        if not (self.source_img and self.target_img):
            QMessageBox.information(self, "Batch Failed", "Please load both images first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if not folder:
            return

        out_dir = os.path.join(folder, "Remapped")
        os.makedirs(out_dir, exist_ok=True)

        self.sync_palettes()
        mapping_hex = {
            rgb_to_hex(t): rgb_to_hex(s)
            for t,s in zip(self.target_palette, self.source_palette)
        }
        payload = {
            "clusters":   self.spin.value(),
            "blur":       self.blur_amount,
            "dither":     self.chk_dither.isChecked(),
            "antialias":  self.chk_antialias.isChecked(),
            "mapping":    mapping_hex
        }
        with open(os.path.join(out_dir, "mapping.json"), 'w', encoding='utf-8') as mf:
            json.dump(payload, mf, indent=4)

        count = 0
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png",".jpg",".jpeg")):
                continue
            try:
                img = Image.open(os.path.join(folder, fname)).convert('RGBA')
            except:
                continue

            self.target_img = img
            self.compute_result()
            final = self.result_img

            name, ext = os.path.splitext(fname)
            final.save(os.path.join(out_dir, f"{name}_remapped{ext}"))
            count += 1

        QMessageBox.information(
            self, "Batch Completed",
            f"Processed {count} images.\nOutputs in: {out_dir}"
        )


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
