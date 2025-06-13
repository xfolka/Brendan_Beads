import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .generate_data import (
    generate_and_visualize_future,
    get_metada_pixelsizes,
    read_ome_tiff,
)

if TYPE_CHECKING:
    import napari


class MainWidget(QWidget):

    show_all_layers_signal = Signal(np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray)
    close_progress_signal = Signal()
    
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.loadImageBtn = QPushButton("Load image")
        self.loadImageBtn.clicked.connect(self._on_load_image)

        self.surface_combo = QComboBox()

        self.beads_combo = QComboBox()

        self.readDataBtn = QPushButton("Start visualization")
        self.readDataBtn.clicked.connect(self._on_start_computing)
        self.readDataBtn.setEnabled(False)

        self.progress_dialog = QProgressDialog("", "", 0, 0)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.reset()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.loadImageBtn)
        self.layout().addWidget(QLabel("Choose surface layer"))
        self.layout().addWidget(self.surface_combo)
        self.layout().addWidget(QLabel("Choose beads layer"))
        self.layout().addWidget(self.beads_combo)
        self.layout().addWidget(self.readDataBtn)

        self.dataFile = None
        self.image_path: Path | None = None

        self.show_all_layers_signal.connect(self._show_all_layers,Qt.QueuedConnection)
        self.close_progress_signal.connect(self._close_progress_dialog, Qt.QueuedConnection)

    def _on_load_image(self):
        data = QFileDialog.getOpenFileName()
        if not data[0]:
            return
        
        if len(self.viewer.layers) > 0 and QMessageBox.question(None, "Clear data?", "All layers will be cleared before loading image. OK?") != QMessageBox.Yes:
            return
        
        self.viewer.layers.clear()
        self.beads_combo.clear()
        self.surface_combo.clear()
        self.image_path = Path(data[0])
        img, mdata = read_ome_tiff(self.image_path)
        psizes = get_metada_pixelsizes(mdata)
        xs = psizes['PhysicalSizeX']
        ys = psizes['PhysicalSizeY']
        zs = psizes['PhysicalSizeZ']
        self.metadata = mdata
        name_generator = (f"channel {x}" for x in itertools.count())
        
        self.viewer.add_image(img, metadata=mdata, channel_axis=0, name=name_generator, scale=(zs, ys, xs))
        for ll in self.viewer.layers:
            self.beads_combo.addItem(ll.name)
            self.surface_combo.addItem(ll.name)
        
        self.readDataBtn.setEnabled(True)

    def _on_start_computing(self):
        if self.image_path is not None:
            self.run_all(self.image_path)

    def _show_all_layers(self, i, ii, s, si, b, bi, v, vi, scale):

        self.viewer.add_image(i, name=ii, scale=scale)
        self.viewer.add_labels(s, name=si, scale=scale)
        self.viewer.add_labels(b, name=bi, scale=scale)
        self.viewer.add_vectors(v, name=vi, edge_color='lime', scale=scale)

        #set 3d view mode
        self.viewer.dims.ndisplay = 3

    def _get_layer_by_name(self, name: str):
        for ll in self.viewer.layers:
            if ll.name == name:
                return ll
            
        return None

    def run_all(self, file: Path):
        surface = self._get_layer_by_name(self.surface_combo.currentText())
        beads = self._get_layer_by_name(self.beads_combo.currentText())

        if surface is None or beads is None:
            return QMessageBox.warning(None, "No data", "No layers to generate data from!")
        
        future = generate_and_visualize_future(beads.data, surface.data, self.metadata, file)
        
        f_cb = self._future_done_callback
        future.add_done_callback(f_cb)
        self.show_progress_dialog("Doing calculation ...", "cancel")

    def _future_done_callback(self, future, success_msg=None, error_msg=None, show_exception=True):
        self.close_progress_signal.emit()

        msg = success_msg
        title = "Success"
        # The task is completed
        if future.done():
            if future.exception() is not None and error_msg is not None:
                msg = error_msg
                title = "Error during task"
                if show_exception:
                    msg += ": " + str(future.exception())
                
            else:
                self.show_all_layers_signal.emit(*future.result())
                self.close_progress_signal.emit()
 
    def _close_progress_dialog(self):
        self.progress_dialog.reset()

    def show_progress_dialog(self,label,cancel):
        self.progress_dialog.setLabelText(label)
        self.progress_dialog.setCancelButtonText(cancel)
        self.progress_dialog.open()
