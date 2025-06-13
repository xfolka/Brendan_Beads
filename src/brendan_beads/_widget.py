from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QFileDialog, QProgressDialog, QLabel, QSpinBox, QMessageBox
from .generate_data import load_and_generate_data, generate_and_visualize_future, read_ome_tiff
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    import napari


class MainWidget(QWidget):


    Z_AXIS = 1
    C_AXIS = 0

    show_all_layers_signal = Signal(np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, int)
    close_progress_signal = Signal()
    
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.loadImageBtn = QPushButton("Load image")
        self.loadImageBtn.clicked.connect(self._on_load_image)

        self.z_spin = QSpinBox()
        self.z_spin.setMinimum(0)
        self.z_spin.setReadOnly(True)

        self.c_spin = QSpinBox()
        self.c_spin.setMinimum(0)
        self.c_spin.setReadOnly(True)

        self.readDataBtn = QPushButton("Start visualization")
        self.readDataBtn.clicked.connect(self._on_start_computing)
        self.readDataBtn.setEnabled(False)

        self.progress_dialog = QProgressDialog("","",0,0)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.reset()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.loadImageBtn)
        self.layout().addWidget(QLabel("Selected Z"))
        self.layout().addWidget(self.z_spin)
        # self.layout().addWidget(QLabel("Selected C"))
        # self.layout().addWidget(self.c_spin)

        self.layout().addWidget(self.readDataBtn)
        self.dataFile = None

        self.show_all_layers_signal.connect(self._show_all_layers,Qt.QueuedConnection)
        self.close_progress_signal.connect(self._close_progress_dialog, Qt.QueuedConnection)
        #viewer.layers.events.inserted.connect(self._image_loaded)
        viewer.dims.events.current_step.connect(self._current_z_or_c_changed)
        
        self.image_path: Path | None = None
        
#        viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

    # def _on_active_layer_changed(self, event):
    #     active_layer = event.value  # The newly selected layer (or None)
    #     print(f"Selected layer changed to: {active_layer}")

    def _current_z_or_c_changed(self, event):

        self.z_spin.setValue(event.value[MainWidget.Z_AXIS])
        self.c_spin.setValue(event.value[MainWidget.C_AXIS])
        #print("jek")

    def _on_load_image(self):
        data = QFileDialog.getOpenFileName()
        if not data[0]:
            return
        self.image_path = Path(data[0])
        img, mdata = read_ome_tiff(self.image_path)
        
        self.viewer.add_image(img, metadata=mdata)
        
        self.readDataBtn.setEnabled(True)
        z = img.shape[MainWidget.Z_AXIS] if img.ndim == 4 else 0
        self.z_spin.setMaximum(z)

    def _image_loaded(self, event):
        #print(event)
        if self.image_path is not None:
            return QMessageBox.warning(None, "Use only one file", "This plugin will only use the first file loaded")
            
        self.image_path = Path(event.value.source.path)
        if event.source.ndim == 4: #change to event.source.ndim
            z = event.value.data.shape[MainWidget.Z_AXIS]
        else:
            z = 0
        c = event.value.data.shape[MainWidget.C_AXIS]

        self.z_spin.setMaximum(z)
        self.c_spin.setMaximum(c)
        self.readDataBtn.setEnabled(True)

    def _on_start_computing(self):
        if self.image_path is not None:
            self.run_all(self.image_path)

    def _show_all_layers(self, i, ii, s, si, b, bi, v, vi, scale):

        self.viewer.add_image(i, name=ii)
        self.viewer.add_labels(s, name=si)
        self.viewer.add_labels(b, name=bi)
        self.viewer.add_vectors(v, name=vi, edge_color='lime')

        #set 3d view mode
        self.viewer.dims.ndisplay = 3

    def run_all(self, file: Path):
        img_data = self.viewer.layers[0].data
        metadata = self.viewer.layers[0].metadata
        z = self.z_spin.value()
        c = self.c_spin.value()

        future = generate_and_visualize_future(img_data[:, z, :, :], metadata, file)
        
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
