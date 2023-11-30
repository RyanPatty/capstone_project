import ctypes
import os
import sys
from enum import Enum

import cv2
import qdarktheme
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QComboBox, QVBoxLayout, QLabel, QProgressBar, QPushButton, \
    QFileDialog, QHBoxLayout, QWidget, QScrollArea, QMainWindow

from VideoProcessor import VideoProcessor


class FrameSkip(Enum):
    """Enum to represent the number of frames to skip between processed frames."""
    ONE_FRAME = 1
    FIVE_FRAMES = 5
    TEN_FRAMES = 10
    FIFTEEN_FRAMES = 15
    ALL_FRAMES = 0


class ProgressWidget(QWidget):
    """Class to display status of video processing of a single file.

    Attributes:
        label (QLabel): Label to display the file name.
        progress_bar (QProgressBar): Progress bar to display the processing progress.
        progress_label (QLabel): Label to display the processing progress as a percentage.

    """

    def __init__(self, label_text, max_value, parent=None):
        super(ProgressWidget, self).__init__()
        self.label = QLabel(label_text, self)
        self.progress_label = QLabel("Not Processed", self)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.frame = QLabel(self)
        self.frame.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setMaximum(max_value)
        self.frame_skip_label = QLabel("Frames per second to process: ", self)
        self.frame_skip_selector = self.getSelector()
        layout = QVBoxLayout()
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.frame_skip_label)
        horizontal_layout.addWidget(self.frame_skip_selector)
        self.setGeometry(0, 0, 400, 200)
        layout.addWidget(self.label)
        layout.addWidget(self.frame)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addLayout(horizontal_layout)
        self.setLayout(layout)
        self.setMaximumHeight(150)

    def getSelector(self):
        dropdown = QComboBox(self)
        dropdown.wheelEvent = lambda event: None
        dropdown.addItem("1 frame", FrameSkip.ONE_FRAME)
        dropdown.addItem("5 frames", FrameSkip.FIVE_FRAMES)
        dropdown.addItem("10 frames", FrameSkip.TEN_FRAMES)
        dropdown.addItem("15 frames", FrameSkip.FIFTEEN_FRAMES)
        dropdown.addItem("All frames", FrameSkip.ALL_FRAMES)
        return dropdown

    def set_progress(self, value, label):
        self.progress_bar.setValue(int(value))
        self.progress_label.setText(label)

    def get_frame_skip_seconds(self):
        return FrameSkip(self.frame_skip_selector.currentData()).value

    def set_frame(self, frame, roi_dimensions=None):
        """Display a frame in the progress dialog.

        Args:
            frame (np.array): Frame to be displayed.
            roi_dimensions (tuple, optional): Dimensions of the ROI to be cropped from the frame. Defaults to None.
        """
        if roi_dimensions:
            left, top, right, bottom = roi_dimensions
            frame = frame[top:bottom, left:right]

        # Resize the frame for display purposes.
        max_display_size = (200, 150)
        self.setMaximumHeight(max_display_size[1] + 100)
        frame = cv2.resize(frame, max_display_size)

        # Convert the frame to a QPixmap and set it to the label.
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.frame.setPixmap(pixmap)


class MainWindow(QMainWindow):
    """Class to handle the main window of the application.

    Attributes:
        central_widget (QWidget): Central widget of the main window.
        central_widget_layout (QVBoxLayout): Layout of the central widget.
        scroll_area (QScrollArea): Scroll area to display the progress widgets.
        scroll_area_widget (QWidget): Widget to display the progress widgets.
        scroll_area_layout (QVBoxLayout): Layout of the scroll area widget.
        load_button (QPushButton): Button to load video files.
        process_button (QPushButton): Button to process video files.
        files (list): List of video files to process.
        widget_list (list): List of progress widgets.
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Spectrum Analyzer Tool")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.central_widget_layout = QVBoxLayout()

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
        self.scroll_area_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_area_widget)

        self.load_button = QPushButton("Load Files", self)
        self.load_button.clicked.connect(self.load_files)
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        self.central_widget_layout.addWidget(self.scroll_area)
        self.central_widget_layout.addWidget(self.load_button)
        self.central_widget_layout.addWidget(self.exit_button)
        self.central_widget.setLayout(self.central_widget_layout)
        self.files = []
        self.widget_list = []

    def add_progress_widget(self, label_text, max_value):
        """function to add a progress widget to the scroll area.

        Args:
            label_text (str): Text to display on the progress widget.
            max_value (int): Maximum value of the progress bar.
        """
        progress_widget = ProgressWidget(label_text, max_value)
        self.widget_list.append(progress_widget)
        self.scroll_area_layout.addWidget(progress_widget, stretch=1)

    def load_files(self):
        """Function to add video files to the list of files to process and create progress widgets for them.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        files, _ = file_dialog.getOpenFileNames(self, "Select video files", "", "MP4 files (*.mp4)")

        if files:
            for file in files:
                self.files.append(file)
                label_text = os.path.basename(file)
                self.add_progress_widget(label_text, 100)
            self.central_widget_layout.removeWidget(self.exit_button)
            self.horizontal_layout = QHBoxLayout()
            self.process_button = QPushButton("Process Files", self)
            self.process_button.clicked.connect(self.process_files)

            self.horizontal_layout.addWidget(self.process_button)
            self.horizontal_layout.addWidget(self.exit_button)
            self.central_widget_layout.addLayout(self.horizontal_layout)

    def process_files(self):
        """Function to process the video files."""
        for index, widget in enumerate(self.widget_list):
            widget.set_progress(0, "Running Optical Character Recognition")
            QApplication.processEvents()
            video_processor = VideoProcessor(self.files[index], widget.get_frame_skip_seconds(), widget)
            video_processor.process()
            widget.set_progress(100, "Done")


import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    app_icon = QtGui.QIcon()
    app_icon.addFile('./icons/icon16.png', QtCore.QSize(16, 16))
    app_icon.addFile('./icons/icon24.png', QtCore.QSize(24, 24))
    app_icon.addFile('./icons/icon32.png', QtCore.QSize(32, 32))
    app_icon.addFile('./icons/icon48.png', QtCore.QSize(48, 48))
    app_icon.addFile('./icons/icon.png', QtCore.QSize(256, 256))
    if os.name == 'nt':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SpectrumAnalyzer.Team3.Version0.9")
        print("Windows OS detected")
    app.setWindowIcon(app_icon)
    qdarktheme.enable_hi_dpi()
    qdarktheme.setup_theme()
    main_window.show()
    sys.exit(app.exec_())