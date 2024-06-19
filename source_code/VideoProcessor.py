import csv
import os
import re
import cv2
import easyocr
import numpy as np
from PyQt5.QtWidgets import QApplication


# Frame processing functions.
def process_frame(frame):
    """Preprocess the video frame for further analysis.

    Args:
        frame (np.array): Input video frame.

    Returns:
        tuple: Grayscale, cropped, and threshold versions of the frame.
    """
    # Crop the frame.
    width = frame.shape[1]
    cropped_frame = frame[:, int(width * 0.25):]

    # Reduce noise with Gaussian blur.
    blurred_frame = cv2.GaussianBlur(cropped_frame, (5, 5), 0)

    # Convert to grayscale for contour detection.
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_frame = clahe.apply(gray_frame)

    # Apply adaptive thresholding.
    adaptive_thresh = cv2.adaptiveThreshold(enhanced_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11,
                                            2)
    return gray_frame, cropped_frame, adaptive_thresh


# Line detection and merging functions.
def detect_lines(frame, orientation):
    """Detect horizontal or vertical lines in a frame using the Hough transform.

    Args:
        frame (np.array): Input frame.
        orientation (str): Desired line orientation, either 'horizontal' or 'vertical'.

    Returns:
        list: Detected lines.
    """
    lines = cv2.HoughLinesP(frame, 1, np.pi / 180, threshold=50, minLineLength=150, maxLineGap=5)

    if orientation == 'horizontal':
        return [line for line in lines if abs(line[0][1] - line[0][3]) < 20] if lines is not None else []
    else:  # 'vertical'
        return [line for line in lines if abs(line[0][0] - line[0][2]) < 20] if lines is not None else []


def merge_lines(lines, orientation='horizontal'):
    """Merge closely spaced lines into a single line.

    Args:
        lines (list): Detected lines.
        orientation (str): Line orientation, either 'horizontal' or 'vertical'.

    Returns:
        np.array: Merged lines.
    """
    if not len(lines):
        return []

    if orientation == 'horizontal':
        def sort_key(x):
            return x[0][1]

        merge_threshold = 10
    else:  # 'vertical'
        def sort_key(x):
            return x[0][0]

        merge_threshold = 10

    sorted_lines = sorted(lines, key=sort_key)
    merged = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        last = merged[-1]
        if abs(sort_key(line) - sort_key(last)) < merge_threshold:
            new_line = [(last[0][0] + line[0][0]) // 2, (last[0][1] + line[0][1]) // 2, (last[0][2] + line[0][2]) // 2,
                        (last[0][3] + line[0][3]) // 2]
            merged[-1] = [new_line]
        else:
            merged.append(line)
    return np.array(merged)


# Functions to handle missing lines and grid detection.
def infer_missing_lines(lines, orientation='horizontal', expected_count=10):
    """Infer missing lines to complete a grid.

    Args:
        lines (list): Detected lines.
        orientation (str): Line orientation, either 'horizontal' or 'vertical'.
        expected_count (int): Expected number of lines for a complete grid.

    Returns:
        np.array: Lines including inferred ones.
    """
    if not lines.size:
        return lines

    if orientation == 'horizontal':
        def sort_key(x):
            return x[0][1]
    else:
        def sort_key(x):
            return x[0][0]

    sorted_lines = sorted(lines, key=sort_key)
    diffs = [sort_key(sorted_lines[i + 1]) - sort_key(sorted_lines[i]) for i in range(len(sorted_lines) - 1)]
    avg_diff = int(np.mean(diffs))
    inferred_lines = sorted_lines.copy()

    while len(inferred_lines) < expected_count:
        first_line = inferred_lines[0]
        inferred_start = sort_key(first_line) - avg_diff
        if orientation == 'horizontal':
            inferred_lines.insert(0, [[first_line[0][0], inferred_start, first_line[0][2], inferred_start]])
        else:
            inferred_lines.insert(0, [[inferred_start, first_line[0][1], inferred_start, first_line[0][3]]])

        if len(inferred_lines) < expected_count:
            last_line = inferred_lines[-1]
            inferred_end = sort_key(last_line) + avg_diff
            if orientation == 'horizontal':
                inferred_lines.append([[last_line[0][0], inferred_end, last_line[0][2], inferred_end]])
            else:
                inferred_lines.append([[inferred_end, last_line[0][1], inferred_end, last_line[0][3]]])
    return np.array(inferred_lines)


def get_grid_bounding_box(horizontal_lines, vertical_lines):
    """Determine the bounding box of a grid based on detected lines.

    Args:
        horizontal_lines (list): Detected horizontal lines.
        vertical_lines (list): Detected vertical lines.

    Returns:
        tuple: Bounding box coordinates (left, top, right, bottom).
    """
    if not horizontal_lines.size and not vertical_lines.size:
        return 0, 0, 0, 0

    inferred_horizontal_lines = infer_missing_lines(horizontal_lines, 'horizontal')
    inferred_vertical_lines = infer_missing_lines(vertical_lines, 'vertical')

    left = min([line[0][0] for line in inferred_vertical_lines])
    right = max([line[0][2] for line in inferred_vertical_lines])
    top = min([line[0][1] for line in inferred_horizontal_lines])
    bottom = max([line[0][3] for line in inferred_horizontal_lines])
    return left, top, right, bottom


# Signal detection functions.
def detect_signal(roi):
    """Detect a signal within a region of interest (ROI).

    Args:
        roi (np.array): Input region of interest.

    Returns:
        tuple: Signal contour and offsets (y_offset, x_offset).
    """
    roi_height = roi.shape[0]
    roi_width = roi.shape[1]

    cropped_roi = roi[int(roi_height * 0.21):int(roi_height * 0.87), int(roi_width * 0.13):int(roi_width * 0.78)]
    y_offset = int(roi_height * 0.21)
    x_offset = int(roi_width * 0.13)

    gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_roi = clahe.apply(gray)

    _, thresh = cv2.threshold(enhanced_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    vertical_structure = np.ones((1, 6), np.uint8)
    eroded_vertical = cv2.erode(thresh, vertical_structure, iterations=1)
    dilated_vertical = cv2.dilate(eroded_vertical, vertical_structure, iterations=1)
    horizontal_structure = np.ones((6, 1), np.uint8)
    eroded_horizontal = cv2.erode(dilated_vertical, horizontal_structure, iterations=1)
    dilated_horizontal = cv2.dilate(eroded_horizontal, horizontal_structure, iterations=1)

    filtered = cv2.bilateralFilter(dilated_horizontal, 9, 75, 75)

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, y_offset, x_offset

    signal_contour = max(contours, key=cv2.contourArea)
    return signal_contour, y_offset, x_offset


class VideoProcessor:
    """Class to handle the processing of video files to detect signal peaks.

    Attributes:
        file_path (str): Path to the video file.
        frame_skip_seconds (int): Number of seconds to skip between processed frames.
        progress_dialog (object): Dialog to show processing progress.
        power_records (list): List to store detected power records.
        initial_power (float): Initial detected power.
        locked_roi (tuple): Bounding box of the Region of Interest.
        frame_counter (int): Counts processed frames.
    """

    def __init__(self, file_path, frame_skip_seconds, progress_dialog):
        self.file_path = file_path
        self.frame_skip_seconds = frame_skip_seconds
        self.progress_dialog = progress_dialog
        self.power_records = []
        self.initial_power = None
        self.locked_roi = None  # Region of Interest
        self.frame_counter = 0  # Counts the number of frames processed
        self.center_frequency = 1.0  # Default center frequency
        self.frequency_resolution = .01  # Default frequency resolution defined by span / 10
        self.max_power = 0  # default RL value
        self.power_resolution = 10  # default Atten value

    def process(self):
        """Main method to process the video."""
        cap = cv2.VideoCapture(self.file_path)
        firstFrame = True
        # Retrieve video properties.
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_skip = 0
        frames_processed = 0
        if self.frame_skip_seconds != 0:
            frames_to_skip = int(fps / self.frame_skip_seconds)
        frames_for_2_seconds = 2 * fps

        if not cap.isOpened():
            print(f"Error: Unable to open video at {self.file_path}.")
            return

        for frame_number in range(total_frames):
            ret, current_frame = cap.read()
            if not ret or current_frame is None:
                break
            if firstFrame:
                firstFrame = False
                self.OCR(current_frame, show=False)

            self.frame_counter += 1

            # Skip certain frames based on the counter.
            if (self.frame_counter <= frames_for_2_seconds and self.frame_counter % 3 != 0) or \
                    (self.frame_counter > frames_for_2_seconds and (
                            frames_to_skip != 0 and self.frame_counter % frames_to_skip != 0)):
                continue
            else:
                frames_processed += 1
            # Update the progress dialog.
            self.progress_dialog.set_progress((self.frame_counter / total_frames) * 100, f"Processing frame "
                                                                                         f"{self.frame_counter}/{total_frames} of video.")
            rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            self.progress_dialog.set_frame(rgb_frame)

            QApplication.processEvents()

            gray_frame, cropped_frame, adaptive_thresh = process_frame(current_frame)

            # If no Region of Interest (ROI) is locked, detect it. Otherwise, use the existing one.
            if self.locked_roi is None:
                horizontal_lines = detect_lines(adaptive_thresh, 'horizontal')
                merged_horizontal_lines = merge_lines(horizontal_lines, 'horizontal')
                vertical_lines = detect_lines(adaptive_thresh, 'vertical')
                merged_vertical_lines = merge_lines(vertical_lines, 'vertical')
                self.locked_roi = get_grid_bounding_box(merged_horizontal_lines, merged_vertical_lines)
            left, top, right, bottom = self.locked_roi

            roi = cropped_frame[top:bottom, left:right]
            if roi.size != 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                self.detect_signal_peak(roi, self.locked_roi, cap)

        print(f"Processed {frames_processed} frames out of {total_frames} frames in the video.")
        self._write_power_records_to_csv()
        cap.release()
        cv2.destroyAllWindows()

    def _write_power_records_to_csv(self):
        """Save detected power records to a CSV file."""
        video_name = os.path.basename(self.file_path)
        video_name_without_extension = os.path.splitext(video_name)[0]
        with open(f'power_records_{video_name_without_extension}.csv', 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'frequency', 'power', 'min_power', 'max_power', 'avg_power']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.power_records:
                writer.writerow({k: "{:.2f}".format(v) if isinstance(v, float) else v for k, v in record.items()})

    def detect_signal_peak(self, roi, grid_dimensions, cap):
        """Detect the peak of a signal within a Region of Interest (ROI).

        Args:
            roi (np.array): Input region of interest.
            grid_dimensions (tuple): Bounding box of the grid.
            cap (cv2.VideoCapture): Video capture object.
        """
        signal_contour, y_offset, x_offset = detect_signal(roi)
        if signal_contour is None:
            return None, None, None, None

        contour_np_array = np.array(signal_contour)
        contour_points = contour_np_array.squeeze()

        top_point = min(contour_points, key=lambda p: p[1])

        cv2.circle(roi, (top_point[0] + x_offset, top_point[1] + y_offset), 5, (0, 165, 255), -1)

        x_center = grid_dimensions[2] // 2
        grid_width = grid_dimensions[2] - grid_dimensions[0]
        freq_units_from_center = (top_point[0] - x_center) / (grid_width / 10)
        frequency = freq_units_from_center * self.frequency_resolution + (self.center_frequency + .02)

        grid_height = grid_dimensions[3] - grid_dimensions[1]
        relative_position = (grid_dimensions[3] - top_point[1]) / grid_height
        power = float(
            (self.max_power - 10 * self.power_resolution) + relative_position * (10 * self.power_resolution) - 13)

        if self.initial_power is None:
            self.initial_power = power
        elif power > self.initial_power:
            video_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            minutes, milliseconds = divmod(video_time_msec, 60000)
            seconds = milliseconds // 1000
            timestamp = f"{int(minutes):02d}:{int(seconds):02d}"

            self.power_records.append({
                'timestamp': timestamp,
                'frequency': float(frequency),
                'power': float(power),
                'min_power': float(min([record['power'] for record in self.power_records] + [power])),
                'max_power': float(max([record['power'] for record in self.power_records] + [power])),
                'avg_power': float(sum([record['power'] for record in self.power_records] +
                                       [power]) / (len(self.power_records) + 1))
            })

    def OCR(self, frame, show=False):
        """Detect Spectrometer parameters using OCR.

        Args:
            frame (np.array): Input video frame.
            show (bool): Display the frame with detected parameters.
        """
        reader = easyocr.Reader(['en'])
        display = []
        image = cv2.copyMakeBorder(frame, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        results = reader.readtext(image, batch_size=3, width_ths=1.5)
        for (bbox, text, prob) in results:
            text = text.lower()
            text = re.sub(r'[o@]', "0", text)
            text = re.sub(r'[i]', "1", text)
            text = re.sub(r'[,:]', ".", text)
            attenMatch = re.search(r'atten\s*(\d+\.?\d*)', text)
            rlMatch = re.search(r'rl\s*(\d+\.?\d*)', text)
            centerMatch = re.search(r'center\s*(\d+\.\d*)', text)
            spanMatch = re.search(r'span\s*(\d+\.\d*)', text)
            if (attenMatch):
                display.append((text, bbox))
                self.power_resolution = int(attenMatch.group(1))
            elif (rlMatch):
                display.append((text, bbox))
                self.max_power = int(rlMatch.group(1))
            elif (centerMatch):
                display.append((text, bbox))
                self.center_frequency = float(centerMatch.group(1))
            elif (spanMatch):
                display.append((text, bbox))
                self.frequency_resolution = float(spanMatch.group(1)) / 10000
        if (show):
            for label, cordinates in display:
                image = cv2.rectangle(image, cordinates[0], cordinates[2], [0, 0, 255], 3)
                cv2.putText(image, text=label, org=(cordinates[2][0] + 5, cordinates[2][1]),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.7, color=[0, 0, 255], thickness=2)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()