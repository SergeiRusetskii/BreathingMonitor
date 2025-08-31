"""Main GUI module for the Breathing Monitor application.

This module defines the :class:`BreathingMonitorApp` which builds the
Tkinter interface, manages video capture and processing threads, and handles
user interaction such as patient selection and calibration.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time
import os
import sys

from monitor_threads import (
    SharedData,
    VideoCaptureThread,
    VideoProcessThread,
)






class BreathingMonitorApp:
    """Main application window managing UI and processing threads."""

    def __init__(self, root):
        """Initialize the Breathing Monitor application."""
        self.root = root
        self.root.title("Breathing Monitor v4.0")

        # Default values
        self.sample_rate = self.graph_window_sec = 1
        self.camera_url = ''
        self.distance_correction = 1

        # Load settings
        self.load_settings()

        # Patient display label above the graph
        self.patient_name = "No patient selected"
        self.patient_label = tk.Label(
            root, 
            text=self.patient_name, 
            font=("Helvetica", 16, "bold"))
        self.patient_label.pack(side=tk.TOP, pady=5)

        # Initialize variables
        self.capture_interval = 1.0 / self.sample_rate
        self.max_length = round(self.graph_window_sec * 1.25) * self.sample_rate
        self.update_interval = int(1000 / self.sample_rate)  # Convert to milliseconds
        self.max_data_points = int(self.graph_window_sec * self.sample_rate)
        self.x_vals = deque(maxlen=self.max_data_points)
        self.y_vals = deque(maxlen=self.max_data_points)
        self.start_time = None
        self.shared_data = SharedData()
        self.capture_thread = None
        self.process_thread = None
        self.cap = cv2.VideoCapture(self.camera_url)
        self.frame_counter = 0

        # Initialize y_shift
        self.y_shift = 0

        self.last_capture_time = 0
        self.adjust_window = None
        self.markers_video_running = False

        # Calibration variables
        self.y_vals_calibrated = self.y_vals
        self.calibration_running = False
        self.calibration_window = None
        self.calibration_message = tk.StringVar()

        # Set up the real-time breathing curve plot
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.line, = self.ax.plot(self.x_vals, self.y_vals, 'r', linewidth=2, zorder=10)
        self.ax.set_title('Breathing Curve', fontsize=16)
        self.ax.set_xlabel('Time (seconds)', fontsize=13)
        self.ax.set_ylabel('Breath Amplitude (cm)', fontsize=13)
        self.ax_right = self.ax.secondary_yaxis(
            'right', 
            functions=(lambda y: y / self.calibration_factor, 
                       lambda y: y * self.calibration_factor))
        self.ax_right.set_ylabel('Breath Amplitude (pixels)', fontsize=13)
        self.upper_threshold_line = self.ax.axhline(
            self.upper_threshold, 
            color='g', linestyle='--', zorder=5)
        self.lower_threshold_line = self.ax.axhline(
            self.lower_threshold, 
            color='b', linestyle='--', zorder=5)
        # Add static line at Y=0
        self.zero_line = self.ax.axhline(0, color='lightgrey', linestyle='-', zorder=5)

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side=tk.LEFT, 
            fill=tk.BOTH, expand=1)

        # Buttons and other UI components
        self.common_font = ("Helvetica", 14)
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.button_style = ttk.Style()
        self.button_style.configure('TButton', font=self.common_font)

        # Groups of controls
        self.setup_control_groups()

        self.running = False
        
    def setup_control_groups(self):
        """Create and layout control frames and widgets."""

        # Configure style for LabelFrame
        self.style = ttk.Style()
        self.style.configure(
            'TLabelframe.Label',
            font=self.common_font)

        # Patient Control Frame
        patient_control_frame = ttk.LabelFrame(
            self.button_frame, 
            text="Patient Control")
        patient_control_frame.pack(fill=tk.X, padx=10, pady=2)

        self.open_patient_button = ttk.Button(
            patient_control_frame, 
            text="Open Patient", 
            command=self.patient_open, 
            style='TButton')
        self.open_patient_button.pack(fill=tk.X)

        self.start_button = ttk.Button(
            patient_control_frame, 
            text="Start", 
            command=self.start_capture, 
            style='TButton')
        self.start_button.pack(fill=tk.X)

        self.stop_button = ttk.Button(
            patient_control_frame, 
            text="Stop", 
            command=self.stop, 
            style='TButton')
        self.stop_button.pack(fill=tk.X)

        self.baseline_button = ttk.Button(
            patient_control_frame, 
            text="Baseline", 
            command=self.baseline, 
            style='TButton')
        self.baseline_button.pack(fill=tk.X)

        # Thresholds Frame
        thresholds_frame = ttk.LabelFrame(
            self.button_frame, text="Thresholds")
        thresholds_frame.pack(fill=tk.X, padx=10, pady=2)
        thresholds_frame.columnconfigure(1, weight=1)

        tk.Label(
            thresholds_frame, 
            text="Upper threshold (cm):", 
            font=self.common_font).grid(row=0, column=0, sticky='w')
        self.upper_threshold_var = tk.DoubleVar(value=self.upper_threshold)
        self.upper_threshold_entry = ttk.Entry(
            thresholds_frame, 
            textvariable=self.upper_threshold_var, 
            width=6, 
            font=self.common_font)
        self.upper_threshold_entry.grid(
            row=0, 
            column=1, 
            sticky='e', 
            padx=(0, 5))

        tk.Label(
            thresholds_frame, 
            text="Lower threshold (cm):", 
            font=self.common_font).grid(row=1, column=0, sticky='w')
        self.lower_threshold_var = tk.DoubleVar(value=self.lower_threshold)
        self.lower_threshold_entry = ttk.Entry(
            thresholds_frame, 
            textvariable=self.lower_threshold_var, 
            width=6, 
            font=self.common_font)
        self.lower_threshold_entry.grid(
            row=1, 
            column=1, 
            sticky='e', 
            padx=(0, 5))

        tk.Label(
            thresholds_frame, 
            text="Couch index:", 
            font=self.common_font).grid(row=2, column=0, sticky='w')
        self.couch_index_var = tk.DoubleVar(value=0.0)
        self.couch_index_entry = ttk.Entry(
            thresholds_frame, 
            textvariable=self.couch_index_var, 
            width=6, 
            font=self.common_font)
        self.couch_index_entry.grid(
            row=2, 
            column=1, 
            sticky='e', 
            padx=(0, 5))

        self.update_thresholds_button = ttk.Button(
            thresholds_frame, 
            text="Update thresholds", 
            command=self.update_thresholds, 
            style='TButton')
        self.update_thresholds_button.grid(
            row=3, 
            column=0, 
            columnspan=2, 
            sticky='ew')

        # Additional Controls Frame
        additional_controls_frame = ttk.LabelFrame(
            self.button_frame, 
            text="Additional Controls")
        additional_controls_frame.pack(
            fill=tk.X, 
            padx=10, 
            pady=2, 
            expand=True, 
            anchor='s')

        self.calibrate_button = ttk.Button(
            additional_controls_frame, 
            text="Calibrate", 
            command=self.calibrate_ask, 
            style='TButton')
        self.calibrate_button.pack(fill=tk.X)

        self.show_video_button = ttk.Button(
            additional_controls_frame, 
            text="Show Video", 
            command=self.show_adjust_limits_window, 
            style='TButton')
        self.show_video_button.pack(fill=tk.X)

        self.exit_button = ttk.Button(
            additional_controls_frame, 
            text="Exit", 
            command=self.exit_program, 
            style='TButton')
        self.exit_button.pack(fill=tk.X)

    def start_capture(self):
        """Start video capture and data processing."""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.x_vals.clear()
            self.y_vals.clear()
            
            capture_limits = (self.capture_top, self.capture_bottom, self.capture_left, self.capture_right)
            self.capture_thread = VideoCaptureThread(self.camera_url, self.shared_data, capture_limits)
            self.process_thread = VideoProcessThread(self.shared_data)
            
            self.capture_thread.start()
            self.process_thread.start()
            
            self.update()

    def stop(self):
        """Stop video capture and data processing."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.join()
            self.capture_thread = None
        if self.process_thread:
            self.process_thread.stop()
            self.process_thread.join()
            self.process_thread = None

    def update(self):
        """Update the graph with new data from the processing thread."""
        if self.running:
            timestamp, amplitude = self.shared_data.get_processed()
            if timestamp is not None and amplitude is not None:
                self.x_vals.append(timestamp - self.start_time)
                self.y_vals.append(amplitude)
                self.update_graph()
            
            self.root.after(self.update_interval, self.update)

    def patient_open(self):
        """Open window to search and select a patient."""

        self.patient_window = tk.Toplevel(self.root)
        self.patient_window.title("Patient Explorer")

        tk.Label(
            self.patient_window, 
            text="Patient ID1:", 
            font=("Helvetica", 14)).pack(pady=10)
        self.patient_id_entry = ttk.Entry(self.patient_window)
        self.patient_id_entry.config(font=("Helvetica", 14))
        self.patient_id_entry.pack(pady=10)

        button_frame = ttk.Frame(self.patient_window)
        button_frame.pack(pady=10)

        find_button = ttk.Button(
            button_frame, 
            text="Find", 
            command=self.patient_find)
        find_button.pack(side=tk.LEFT, padx=5)

        cancel_button = ttk.Button(
            button_frame, 
            text="Cancel", 
            command=self.patient_window.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

    def update_graph(self):
        """Update the breathing curve graph."""
        if self.x_vals and self.y_vals:
            y_vals_calibrated = [
                (y * self.calibration_factor * self.distance_correction) - 
                (self.y_shift * self.calibration_factor * self.distance_correction) for y in self.y_vals
            ]
            self.line.set_data(self.x_vals, y_vals_calibrated)
            self.line.set_zorder(10)

            if self.x_vals[-1] > self.graph_window_sec:
                self.ax.set_xlim(self.x_vals[-1] - self.graph_window_sec, self.x_vals[-1])
            else:
                self.ax.set_xlim(0, self.graph_window_sec)

            visible_y_vals = y_vals_calibrated[-self.max_data_points:]
            if visible_y_vals:
                y_min, y_max = min(visible_y_vals), max(visible_y_vals)
                # Adjust y-axis max-limit to include upper threshold
                y_max = max(y_max, self.upper_threshold)
                # Adjust y-axis min-limit to include baseline
                y_min = min(y_min, 0)
                margin = (y_max - y_min) * 0.1  # 10% margin
                self.ax.set_ylim(y_min - margin, y_max + margin)

            self.upper_threshold_line.set_ydata([self.upper_threshold, self.upper_threshold])
            self.lower_threshold_line.set_ydata([self.lower_threshold, self.lower_threshold])

            current_y_value = y_vals_calibrated[-1] if y_vals_calibrated else 0
            if self.lower_threshold <= current_y_value <= self.upper_threshold:
                self.ax.set_facecolor('lightgreen')
            else:
                self.ax.set_facecolor('white')

            self.canvas.draw()

    def get_settings_file_path(self):
        """
        Determine the path to the settings.txt file.
        Look in the parent directory of the application's root folder.
        """
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            application_path = os.path.dirname(sys.executable)
        else:
            # Running as a script
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to the parent directory
        parent_dir = os.path.dirname(application_path)
        return os.path.join(parent_dir, "settings.txt")

    def load_settings(self):
        """
        Load settings from the settings.txt file.
        """
        settings_file = self.get_settings_file_path()
        default_settings = {
            "sample_rate": 6,
            "graph_window_sec": 20,
            "capture_top": 650,
            "capture_bottom": 800,
            "capture_left": 950,
            "capture_right": 1150,
            "calibration_factor": 0.7,
            "lower_threshold": 1.7,
            "upper_threshold": 1.8,
            "camera_url": 'rtsp://admin:camera01@192.168.40.70:554/cam/realmonitor?channel=1&subtype=0'
        }

        # Set default values
        for key, value in default_settings.items():
            setattr(self, key, value)

        if os.path.exists(settings_file):
            with open(settings_file, "r") as file:
                for line in file:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key in default_settings:
                            if isinstance(default_settings[key], int):
                                setattr(self, key, int(value))
                            elif isinstance(default_settings[key], float):
                                setattr(self, key, float(value))
                            else:
                                setattr(self, key, value)

    def save_settings(self):
        """
        Save current settings to the settings.txt file.
        """
        settings_file = self.get_settings_file_path()
        settings_to_save = [
            "sample_rate",
            "graph_window_sec",
            "capture_top",
            "capture_bottom",
            "capture_left",
            "capture_right",
            "calibration_factor",
            "camera_url"
        ]
        
        with open(settings_file, "w") as file:
            for key in settings_to_save:
                file.write(f"{key}: {getattr(self, key)}\n")


    def baseline(self):
        """Set the baseline to the minimum value."""
        required_samples = int(self.sample_rate * 10)
        if len(self.y_vals) < required_samples:
            messagebox.showwarning(
                "Warning", 
                "Not enough data. Please record more.")
            return

        # Convert the last 20 seconds of data to a list for min() operation
        last_20_seconds = list(self.y_vals)[-required_samples:]
        self.y_shift = min(last_20_seconds)
        self.update_graph()


    def update_thresholds(self):
        """Apply updated threshold values from the UI fields."""

        upper = self.upper_threshold_var.get()
        lower = self.lower_threshold_var.get()
        index = self.couch_index_var.get()

        # Validate index
        valid_indices = {i/2 for i in range(-2, 9)}  # -1.0, -0.5, 0, 0.5, 1.0, ..., 4.0
        if index not in valid_indices:
            messagebox.showerror(
                "Error", 
                f"Incorrect couch index = {index}")
            return

        if upper <= lower:
            messagebox.showerror(
                "Error", 
                "Upper threshold should be greater than lower!")
            return

        self.upper_threshold = upper
        self.lower_threshold = lower
        self.couch_index = index
        
        # Calculate distance correction
        horizontal = 150 - index * 14
        self.distance_correction = np.sqrt(55**2 + horizontal**2) / 165
        
        self.upper_threshold_line.set_ydata(self.upper_threshold)
        self.lower_threshold_line.set_ydata(self.lower_threshold)
        self.canvas.draw()

    def exit_program(self):
        """Clean up and exit the program."""
        self.stop()
        self.root.quit()
        self.root.destroy()
        cv2.destroyAllWindows()




    def patient_find(self):
        """Find a patient in the CSV database based on ID1."""
        patient_id = self.patient_id_entry.get().strip()
        database_path = r'\\VARIANCOM\\va_transfer\BreathingMonitorDatabase\\Patient_database_new.csv'
        
        try:
            # Check if the file exists and is accessible
            if not os.path.exists(database_path):
                messagebox.showerror("Error", "Patient database file not found or not accessible!")
                return

            # Read CSV file, ensuring ID1 is treated as string
            df = pd.read_csv(database_path, dtype={'ID1': str})
            patient_row = df[df['ID1'].str.strip() == patient_id]
            if not patient_row.empty:
                self.patient_show_details(patient_row.iloc[0])
            else:
                messagebox.showerror(
                    "Error", 
                    f"There is no Patient with ID1={patient_id} in the database!")
        except pd.errors.EmptyDataError:
            messagebox.showerror("Error", "Patient database file is empty!")
        except PermissionError:
            messagebox.showerror("Error", "Access denied to the patient database file. Please check your permissions.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def patient_show_details(self, patient_row):
        """Display patient details in a new window."""
        self.patient_details_window = tk.Toplevel(self.root)
        self.patient_details_window.title("Patient Details")
        info = (
            f"ID1: {patient_row['ID1']}"
            f"\nName: {patient_row['Patient name']}"
            f"\nUpper threshold (cm): {patient_row['Upper threshold (cm)']}"
            f"\nLower threshold (cm): {patient_row['Lower threshold (cm)']}"
            f"\nCouch index: {patient_row['Couch index']}"
        )
        
        info_label = tk.Label(self.patient_details_window, text=info)
        info_label.config(font=("Helvetica", 14))
        info_label.pack(pady=5)
        button_frame = ttk.Frame(self.patient_details_window)
        button_frame.pack(pady=10)
        load_button = ttk.Button(
            button_frame, 
            text="Load data", 
            command=lambda: self.patient_load_data(patient_row))
        load_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(
            button_frame, 
            text="Cancel", 
            command=self.patient_details_window.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)


    def patient_load_data(self, patient_row):
        """Populate fields using the selected patient's record."""

        try:
            self.patient_id = patient_row['ID1']
            self.patient_name = patient_row['Patient name']
            self.lower_threshold = float(patient_row['Lower threshold (cm)'])
            self.upper_threshold = float(patient_row['Upper threshold (cm)'])
            self.couch_index = float(patient_row['Couch index'])
            
            self.lower_threshold_var.set(self.lower_threshold)
            self.upper_threshold_var.set(self.upper_threshold)
            
            # Calculate distance correction
            horizontal = 150 - self.couch_index * 14
            self.distance_correction = np.sqrt(55**2 + horizontal**2) / 165
            
            self.lower_threshold_var.set(self.lower_threshold)
            self.upper_threshold_var.set(self.upper_threshold)
            self.couch_index_var.set(self.couch_index)
            
            self.update_thresholds()
            self.patient_label.config(
                text=f"ID1: {self.patient_id} Name: {self.patient_name}")
            
            # Close both windows after loading data
            if self.patient_details_window:
                self.patient_details_window.destroy()
            if self.patient_window:
                self.patient_window.destroy()

        except ValueError as e:
            messagebox.showerror(
                "Error", 
                f"Value error: {str(e)} - Ensure all values are numeric")


    def show_adjust_limits_window(self):
        """Open a window to adjust video capture boundaries."""

        if not self.adjust_window:
            self.adjust_window = tk.Toplevel(self.root)
            self.adjust_window.title("Adjust Capture Limits")
            self.adjust_window.minsize(300, 200)  # Set minimum size for the window

            entry_width = 10  # Width of the entry fields

            # Numerical fields for Top, Bottom, Left, Right
            tk.Label(
                self.adjust_window, 
                text="Top:", 
                font=("Helvetica", 14)).grid(row=0, column=1)
            self.capture_top_entry = ttk.Entry(
                self.adjust_window, 
                width=entry_width)
            self.capture_top_entry.config(font=("Helvetica", 14))
            self.capture_top_entry.grid(row=1, column=1)
            self.capture_top_entry.insert(0, str(self.capture_top))

            tk.Label(
                self.adjust_window, 
                text="Left:", 
                font=("Helvetica", 14)).grid(row=1, column=0)
            self.capture_left_entry = ttk.Entry(
                self.adjust_window, 
                width=entry_width)
            self.capture_left_entry.config(font=("Helvetica", 14))
            self.capture_left_entry.grid(row=2, column=0)
            self.capture_left_entry.insert(0, str(self.capture_left))

            tk.Label(
                self.adjust_window, 
                text="Right:", 
                font=("Helvetica", 14)).grid(row=1, column=2)
            self.capture_right_entry = ttk.Entry(
                self.adjust_window, 
                width=entry_width)
            self.capture_right_entry.config(font=("Helvetica", 14))
            self.capture_right_entry.grid(row=2, column=2)
            self.capture_right_entry.insert(0, str(self.capture_right))

            tk.Label(
                self.adjust_window, 
                text="Bottom:", 
                font=("Helvetica", 14)).grid(row=2, column=1)
            self.capture_bottom_entry = ttk.Entry(
                self.adjust_window, 
                width=entry_width)
            self.capture_bottom_entry.config(font=("Helvetica", 14))
            self.capture_bottom_entry.grid(row=3, column=1)
            self.capture_bottom_entry.insert(0, str(self.capture_bottom))

            # Update and Close buttons
            update_button = ttk.Button(
                self.adjust_window, 
                text="Update", 
                command=self.update_capture_limits)
            update_button.grid(row=4, column=0, columnspan=3, pady=10)

            close_button = ttk.Button(
                self.adjust_window, 
                text="Close", 
                command=self.close_adjust_limits_window)
            close_button.grid(row=5, column=0, columnspan=3)

            self.adjust_window.protocol(
                "WM_DELETE_WINDOW", 
                self.close_adjust_limits_window)

            self.show_adjust_video()


    def show_adjust_video(self):
        """Start video feed for capture limit adjustments."""

        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Error",
                "Cannot connect to camera video stream!")
            return

        self.video_running = True
        self.update_adjust_video()

    
    def update_adjust_video(self):
        """Update the adjustment window with the latest video frame."""

        if self.video_running:
            ret, img = self.cap.read()
            if ret:
                # Crop the image according to the monitor limits
                img = img[
                    self.capture_top:self.capture_bottom, 
                    self.capture_left:self.capture_right]
                
                # Detect markers
                gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresholded = cv2.threshold(
                    gray_frame, 
                    245, 
                    255, 
                    cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    thresholded, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(
                    contours, 
                    key=cv2.contourArea, 
                    reverse=True)[:4]  # Keep the 4 largest contours
                markers_contours = contours[:4]

                frame_visual = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                for cnt in markers_contours:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        cv2.circle(frame_visual, (cX, cY), 7, (0, 0, 255), -2)

                # Show the adjusted video with markers
                cv2.imshow("Adjust Capture Limits - Video", frame_visual)

                if cv2.getWindowProperty(
                    "Adjust Capture Limits - Video", 
                    cv2.WND_PROP_VISIBLE) < 1:
                    self.video_running = False
                cv2.waitKey(1)
            else:
                messagebox.showerror(
                    "Error", 
                    "Failed to capture video stream.")
            self.root.after(1, self.update_adjust_video)


    def update_capture_limits(self):
        """Store new capture limits from the adjustment window."""

        self.capture_top = int(self.capture_top_entry.get())
        self.capture_bottom = int(self.capture_bottom_entry.get())
        self.capture_left = int(self.capture_left_entry.get())
        self.capture_right = int(self.capture_right_entry.get())
        self.save_settings()
        
        # Update the video feed with new limits
        self.update_capture_area()


    def update_capture_area(self):
        """Ensure the capture object reflects updated limits."""

        # Ensure the capture object is open
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_url)


    def close_adjust_limits_window(self):
        """Close the capture limits adjustment window and cleanup."""

        if self.adjust_window:
            self.video_running = False
            # Close the OpenCV window
            if cv2.getWindowProperty(
                "Adjust Capture Limits - Video",
                cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Adjust Capture Limits - Video")
            self.adjust_window.destroy()
            self.adjust_window = None




    def calibrate_ask(self):
        """Display instructions and ask user to proceed with calibration."""

        dialog = tk.Toplevel(self.root)
        dialog.title("Calibration Procedure")

        tk.Label(dialog, text=(
            "Calibration process:\n"
            "1. Place the breathing phantom in the proper position\n"
            "with the marker block and turn it on.\n"
            "2. Press 'Start' button in the main window\n"
            "and record at least 25 seconds of data.\n"
            "3. Press 'Stop' button and relaunch Calibration.\n\n"
            "Do you want to proceed with the calibration procedure?"
        ), font=("Helvetica", 14)).pack(pady=10)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        yes_button = ttk.Button(
            button_frame, 
            text="Yes", 
            command=lambda: self.calibrate_proceed_with_calibration(dialog))
        yes_button.pack(side=tk.LEFT, padx=5)

        no_button = ttk.Button(
            button_frame, 
            text="Cancel", 
            command=dialog.destroy)
        no_button.pack(side=tk.LEFT, padx=5)

    def calibrate_proceed_with_calibration(self, dialog):
        """Close the prompt and open the calibration window."""

        dialog.destroy()
        self.calibrate_show_calibration_window()

    def calibrate_show_calibration_window(self):
        """Show window used to compute calibration factor."""

        if not self.calibration_window:
            self.calibration_window = tk.Toplevel(self.root)
            self.calibration_window.title("Calibrate Breathing Amplitude")

            tk.Label(
                self.calibration_window, 
                text="Reference Amplitude (cm):", 
                font=("Helvetica", 14)).grid(row=0, column=0)
            self.reference_amplitude = tk.DoubleVar(value=1.8)
            self.reference_entry = ttk.Entry(
                self.calibration_window, 
                textvariable=self.reference_amplitude)
            self.reference_entry.config(font=("Helvetica", 14))
            self.reference_entry.grid(row=0, column=1)

            tk.Label(
                self.calibration_window, 
                text="Measured Amplitude (pixels):", 
                font=("Helvetica", 14)).grid(row=1, column=0)
            self.measured_amplitude = tk.StringVar(value="0")
            self.measured_label = ttk.Label(
                self.calibration_window, 
                textvariable=self.measured_amplitude, 
                font=("Helvetica", 14))
            self.measured_label.grid(row=1, column=1)

            calc_button = ttk.Button(
                self.calibration_window, 
                text="Calculate Amplitude", 
                command=self.calibrate_calculate_amplitude)
            calc_button.grid(row=1, column=2)

            apply_button = ttk.Button(
                self.calibration_window, 
                text="Apply Calibration", 
                command=self.calibrate_apply_calibration)
            apply_button.grid(row=2, column=1)

            close_button = ttk.Button(
                self.calibration_window, 
                text="Close", 
                command=self.calibrate_close_calibration_window)
            close_button.grid(row=2, column=2)

    def calibrate_calculate_amplitude(self):
        """Calculate measured amplitude from collected data."""

        # Ensure at least 20 seconds of data
        if len(self.y_vals) < self.sample_rate * 20:
            messagebox.showerror("Error", "Not enough data captured!")
            return

        amplitude = max(self.y_vals) - min(self.y_vals)
        self.measured_amplitude.set(f"{amplitude:.2f}")

    def calibrate_apply_calibration(self):
        """Apply calibration factor based on measured amplitude."""

        reference_amp = self.reference_amplitude.get()
        measured_amp = float(self.measured_amplitude.get())
        if measured_amp != 0:
            self.calibration_factor = reference_amp / measured_amp
            self.calibration_message.set("Calibration successful")
            self.save_settings()  # Save the calibration factor
            self.update_graph()  # Update the graph to display in cm
        else:
            self.calibration_message.set("Calibration failed")

    def calibrate_close_calibration_window(self):
        """Close calibration window and reset state."""

        if self.calibration_window:
            self.calibration_window.destroy()
            self.calibration_window = None



if __name__ == "__main__":
    root = tk.Tk()
    app = BreathingMonitorApp(root)
    root.mainloop()