from typing import TYPE_CHECKING

from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QFileDialog, QProgressDialog
from .generate_data import load_and_generate_data, generate_and_visualize_future
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    import napari

class MainWidget(QWidget):
    
    
    show_all_layers_signal = Signal( np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, int)
    close_progress_signal = Signal()
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        readDataBtn = QPushButton("Open file to visualize...")
        readDataBtn.clicked.connect(self._on_open_file)

        self.progress_dialog = QProgressDialog("","",0,0)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.reset()
        
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(readDataBtn)
        self.dataFile = None
        
        self.show_all_layers_signal.connect(self._show_all_layers,Qt.QueuedConnection)
        self.close_progress_signal.connect(self._close_progress_dialog, Qt.QueuedConnection)

    def _on_open_file(self):
        data = QFileDialog.getOpenFileName()
        if not data[0]:
            return
        self.dataFile = data[0]
        
        self.run_all(self.dataFile)
        
    def _show_all_layers(self, i,ii , s,si , b,bi , v,vi , scale):
        
        self.viewer.add_image(i, name=ii)
        self.viewer.add_labels(s, name=si)
        self.viewer.add_labels(b, name=bi)
        self.viewer.add_vectors(v, name=vi, edge_color='lime')
        
        #set 3d view mode
        self.viewer.dims.ndisplay = 3
        
    def run_all(self, file: Path):
        future = generate_and_visualize_future(Path(file))
        
        f_cb = self._future_done_callback
        future.add_done_callback(f_cb)
        self.show_progress_dialog("Doing calculation ...","cancel")


    def _future_done_callback(self, future, success_msg = None, error_msg = None, show_exception = True):
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