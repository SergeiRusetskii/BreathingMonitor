import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import font
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time
import os
from itertools import islice
import threading
import queue


class VideoCaptureThread(threading.Thread):
    """Thread class for capturing and processing video frames."""

    def __init__(self, camera_url, capture_limits, sample_rate):
        """
        Initialize the video capture thread.

        :param camera_url: URL of the camera stream
        :param capture_limits: Tuple of (top, bottom, left, right) capture limits
        :param sample_rate: Number of frames to capture per second
        """
        threading.Thread.__init__(self)
        self.camera_url = camera_url
        self.capture_limits = capture_limits
        self.capture_interval = 1.0 / sample_rate
        self.queue = queue.Queue()
        self.stopped = False

    def run(self):
        """Main method to run the thread. Captures and processes frames."""
        cap = cv2.VideoCapture(self.camera_url)
        last_capture_time = time.time()

        while not self.stopped:
            current_time = time.time()
            if current_time - last_capture_time >= self.capture_interval:
                ret, img = cap.read()
                if ret:
                    processed_data = self.process_frame(img, current_time)
                    if processed_data:
                        self.queue.put(processed_data)
                    last_capture_time = current_time
                else:
                    print("Failed to capture frame")
            time.sleep(0.001)  # Short sleep to prevent CPU overuse

        cap.release()

    def process_frame(self, img, timestamp):
        """
        Process a single frame from the video stream.

        :param img: The captured image frame
        :param timestamp: The timestamp of the frame capture
        :return: Tuple of (timestamp, avg_y) or None if processing fails
        """
        top, bottom, left, right = self.capture_limits
        img = img[top:bottom, left:right]

        if img.size == 0:
            return None

        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_frame, 245, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresholded, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) < 2:
            return None

        # Keep the 4 largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        # Select the top 2 contours with the highest Y values
        top_contours = sorted(
            contours,
            key=lambda cnt: min([pt[0][1] for pt in cnt]),
            reverse=False
        )[:2]

        coords = []
        for cnt in top_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                coords.append((cX, cY))

        if len(coords) == 2:
            frame_height = bottom - top
            avg_y = frame_height // 2 - int(sum(y for _, y in coords) / len(coords))
            return (timestamp, avg_y)

        return None

    def stop(self):
        """Stop the thread."""
        self.stopped = True


class BreathingMonitorApp:
    def __init__(self, root):
        """
        Initialize the Breathing Monitor application.
        """
        self.root = root
        self.root.title("Breathing Monitor")

        # Default values
        self.sample_rate = self.graph_window_sec = 1
        self.camera_url = ''

        # Load settings
        self.main_load_settings()

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
        self.x_vals = deque(maxlen=self.max_length)
        self.y_vals = deque(maxlen=self.max_length)
        self.cap = cv2.VideoCapture(self.camera_url)
        self.frame_counter = 0

        # Initialize y_shift
        self.y_shift = 0
        self.running = False
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
        self.line, = self.ax.plot(self.x_vals, self.y_vals, 'r')
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
            color='g', linestyle='--')
        self.lower_threshold_line = self.ax.axhline(
            self.lower_threshold, 
            color='b', linestyle='--')

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

        self.capture_thread = None
        self.data_queue = queue.Queue()        


    def setup_control_groups(self):
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
            command=self.main_patient_open, 
            style='TButton')
        self.open_patient_button.pack(fill=tk.X)

        self.start_button = ttk.Button(
            patient_control_frame, 
            text="Start", 
            command=self.main_start_capture, 
            style='TButton')
        self.start_button.pack(fill=tk.X)

        self.stop_button = ttk.Button(
            patient_control_frame, 
            text="Stop", 
            command=self.main_stop, 
            style='TButton')
        self.stop_button.pack(fill=tk.X)

        self.baseline_button = ttk.Button(
            patient_control_frame, 
            text="Baseline", 
            command=self.main_baseline, 
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

        self.update_thresholds_button = ttk.Button(
            thresholds_frame, 
            text="Update thresholds", 
            command=self.main_update_thresholds, 
            style='TButton')
        self.update_thresholds_button.grid(
            row=2, 
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
            command=self.video_show_adjust_limits_window, 
            style='TButton')
        self.show_video_button.pack(fill=tk.X)

        self.exit_button = ttk.Button(
            additional_controls_frame, 
            text="Exit", 
            command=self.main_exit_program, 
            style='TButton')
        self.exit_button.pack(fill=tk.X)



    def main_start_capture(self):
        """Start the video capture and processing."""
        if not self.running:
            self.running = True
            self.x_vals.clear()
            self.y_vals.clear()
            self.start_time = time.time()
            
            capture_limits = (
                self.capture_top, 
                self.capture_bottom, 
                self.capture_left, 
                self.capture_right
            )
            self.capture_thread = VideoCaptureThread(
                self.camera_url, 
                capture_limits, 
                self.sample_rate
            )
            self.capture_thread.start()
            
            self.main_update()


    def main_stop(self):
        """Stop the video capture and processing."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.join()
            self.capture_thread = None

    
    def main_update(self):
        """Update the graph with new data from the capture thread."""
        if self.running and self.capture_thread:
            while not self.capture_thread.queue.empty():
                timestamp, avg_y = self.capture_thread.queue.get()
                self.x_vals.append(timestamp - self.start_time)
                self.y_vals.append(avg_y)

            self.main_update_graph()
            self.root.after(int(self.capture_interval * 1000), self.main_update)

    def main_patient_open(self):
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


    def main_update_graph(self):
        if self.x_vals and self.y_vals:
            # Apply the calibration factor to y_vals for the left y-axis
            self.y_vals_calibrated = [
                (y * self.calibration_factor) - 
                (self.y_shift * self.calibration_factor) for y in self.y_vals
                ]
            self.line.set_xdata(self.x_vals)
            self.line.set_ydata(self.y_vals_calibrated)

            if self.x_vals[-1] > self.graph_window_sec:
                start_time_index = next(
                    x[0] for x in enumerate(self.x_vals) if 
                    x[1] > self.x_vals[-1] - self.graph_window_sec
                    )
                calibrated_visible_y_vals = list(self.y_vals_calibrated)[start_time_index:]
                self.ax.set_xlim(self.x_vals[start_time_index], self.x_vals[-1])
            else:
                calibrated_visible_y_vals = list(self.y_vals_calibrated)
                self.ax.set_xlim(0, self.graph_window_sec)

            min_calibrated_y_val = min(calibrated_visible_y_vals, default=0)
            max_calibrated_y_val = max(calibrated_visible_y_vals, default=100)

            # Set limits for the left y-axis (calibrated in cm)
            self.ax.set_ylim(min_calibrated_y_val - 1, max_calibrated_y_val + 1)
            self.ax.relim()
            self.ax.autoscale_view()

            # Set limits for the right y-axis (in pixels)
            self.ax_right.set_ylabel('Breath Amplitude (pixels)')

            # Update threshold lines
            self.upper_threshold_line.set_ydata(self.upper_threshold)
            self.lower_threshold_line.set_ydata(self.lower_threshold)

            # Check the current y-value and update background colour
            current_y_value = self.y_vals_calibrated[-1] if self.y_vals_calibrated else 0
            if self.lower_threshold <= current_y_value <= self.upper_threshold:
                self.ax.set_facecolor('lightgreen')
            else:
                self.ax.set_facecolor('white')

            self.canvas.draw()


    def main_load_settings(self):
        """
        Load settings from the settings.txt file.
        """
        settings_file = os.path.join(os.path.dirname(__file__), "settings.txt")
        default_settings = {
            "sample_rate": 6,
            "graph_window_sec": 20,
            "capture_top": 650,
            "capture_bottom": 800,
            "capture_left": 950,
            "capture_right": 1150,
            "calibration_factor": 0.7,
            "lower_threshold": 1.1,
            "upper_threshold": 1.8,
            "camera_url": 'rtsp://admin:camera01@192.168.40.71:554/cam/realmonitor?channel=1&subtype=0'
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


    def main_save_settings(self):
        """
        Save current settings to the settings.txt file.
        """
        settings_file = os.path.join(os.path.dirname(__file__), "settings.txt")
        settings_to_save = [
            "sample_rate",
            "graph_window_sec",
            "capture_top",
            "capture_bottom",
            "capture_left",
            "capture_right",
            "calibration_factor",
            "lower_threshold",
            "upper_threshold",
            "camera_url"
        ]
        
        with open(settings_file, "w") as file:
            for key in settings_to_save:
                file.write(f"{key}: {getattr(self, key)}\n")


    def find_local_minima(self, data, window_size):
        """Find local minima in the data."""
        min_indices = []
        for i in range(window_size, len(data) - window_size):
            window = list(islice(data, i-window_size, i+window_size+1))
            if data[i] == min(window):
                min_indices.append(i)
        return min_indices

    def detect_breathing_cycles(self, min_indices, min_cycle_length):
        """Detect breathing cycles based on local minima."""
        cycles = []
        for i in range(len(min_indices) - 1):
            if min_indices[i+1] - min_indices[i] >= min_cycle_length:
                cycles.append((min_indices[i], min_indices[i+1]))
        return cycles

    def main_baseline(self):
        # Ensure at least 20 seconds of data
        if len(self.y_vals) < self.sample_rate * 20:
            messagebox.showwarning(
                "Warning", 
                "Not enough data for baseline calculation. Please record more data.")
            return

        # Find local minima
        window_size = int(self.sample_rate)  # 1 second window
        min_indices = self.find_local_minima(self.y_vals, window_size)

        # Detect breathing cycles
        # Minimum 2 seconds per cycle
        min_cycle_length = int(self.sample_rate * 2)  
        cycles = self.detect_breathing_cycles(min_indices, min_cycle_length)

        if len(cycles) < 3:
            messagebox.showwarning(
                "Warning", 
                "Not enough complete breathing cycles detected. Please record more data.")
            return

        # Calculate local minima for each of the last 3-5 cycles
        cycle_minima = []
        for start, end in reversed(cycles[-5:]):  # Use up to 5 most recent cycles
            cycle_data = list(islice(self.y_vals, start, end))
            cycle_min = min(cycle_data)
            cycle_minima.append(cycle_min)

        # Calculate average of local minima
        self.y_shift = np.mean(cycle_minima)

        # Update the graph
        self.main_update_graph()

        messagebox.showinfo(
            "Baseline", 
            f"Baseline set using {len(cycle_minima)} most recent breathing cycles.")


    def main_update_thresholds(self):
        upper = self.upper_threshold_var.get()
        lower = self.lower_threshold_var.get()

        if upper <= lower:
            messagebox.showerror(
                "Error", 
                "Upper threshold should be greater than lower!")
        else:
            self.upper_threshold = upper
            self.lower_threshold = lower
            self.upper_threshold_line.set_ydata(self.upper_threshold)
            self.lower_threshold_line.set_ydata(self.lower_threshold)
            self.canvas.draw()

    def main_exit_program(self):
        """Clean up and exit the program."""
        self.main_stop()
        self.root.quit()
        self.root.destroy()
        cv2.destroyAllWindows()




    def patient_find(self):
        """Find a patient in the CSV database based on ID1."""
        patient_id = self.patient_id_entry.get().strip()
        try:
            # Read CSV file, ensuring ID1 is treated as string
            df = pd.read_csv('Patient_database.csv', dtype={'ID1': str})
            patient_row = df[df['ID1'].str.strip() == patient_id]
            if not patient_row.empty:
                self.patient_show_details(patient_row.iloc[0])
            else:
                messagebox.showerror(
                    "Error", 
                    f"There is no Patient with ID1={patient_id} in the database!")
        except FileNotFoundError:
            messagebox.showerror("Error", "Patient database file not found!")
        except pd.errors.EmptyDataError:
            messagebox.showerror("Error", "Patient database file is empty!")
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
        try:
            self.patient_id = patient_row['ID1']
            self.patient_name = patient_row['Patient name']
            self.lower_threshold = float(patient_row['Lower threshold (cm)'])
            self.upper_threshold = float(patient_row['Upper threshold (cm)'])
            
            self.lower_threshold_var.set(self.lower_threshold)
            self.upper_threshold_var.set(self.upper_threshold)
            
            self.main_update_thresholds()  # Update threshold lines
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
                f"Value error: {str(e)} - Ensure the thresholds are numeric")





    def video_show_adjust_limits_window(self):
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
                command=self.video_update_capture_limits)
            update_button.grid(row=4, column=0, columnspan=3, pady=10)

            close_button = ttk.Button(
                self.adjust_window, 
                text="Close", 
                command=self.video_close_adjust_limits_window)
            close_button.grid(row=5, column=0, columnspan=3)

            self.adjust_window.protocol(
                "WM_DELETE_WINDOW", 
                self.video_close_adjust_limits_window)

            self.video_show_adjust_video()


    def video_show_adjust_video(self):
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Error", 
                "Cannot connect to camera video stream!")
            return

        self.video_running = True
        self.video_update_adjust_video()

    
    def video_update_adjust_video(self):
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
                top_contours = sorted(
                    contours, 
                    key=lambda cnt: min([pt[0][1] for pt in cnt]), 
                    reverse=False)[:2]

                frame_visual = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                for cnt in top_contours:
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
            self.root.after(1, self.video_update_adjust_video)


    def video_update_capture_limits(self):
        self.capture_top = int(self.capture_top_entry.get())
        self.capture_bottom = int(self.capture_bottom_entry.get())
        self.capture_left = int(self.capture_left_entry.get())
        self.capture_right = int(self.capture_right_entry.get())
        self.main_save_settings()
        
        # Update the video feed with new limits
        self.video_update_capture_area()


    def video_update_capture_area(self):
        # Ensure the capture object is open
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_url)


    def video_close_adjust_limits_window(self):
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
        dialog.destroy()
        self.calibrate_show_calibration_window()


    def calibrate_show_calibration_window(self):
        if not self.calibration_window:
            self.calibration_window = tk.Toplevel(self.root)
            self.calibration_window.title("Calibrate Breathing Amplitude")

            tk.Label(
                self.calibration_window, 
                text="Reference Amplitude (cm):", 
                font=("Helvetica", 14)).grid(row=0, column=0)
            self.reference_amplitude = tk.DoubleVar(value=1.3)
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
        # Ensure at least 20 seconds of data
        if len(self.y_vals) < self.sample_rate * 20:
            messagebox.showerror("Error", "Not enough data captured!")
            return

        amplitude = max(self.y_vals) - min(self.y_vals)
        self.measured_amplitude.set(f"{amplitude:.2f}")


    def calibrate_apply_calibration(self):
        reference_amp = self.reference_amplitude.get()
        measured_amp = float(self.measured_amplitude.get())
        if measured_amp != 0:
            self.calibration_factor = reference_amp / measured_amp
            self.calibration_message.set("Calibration successful")
            self.main_save_settings()  # Save the calibration factor
            self.main_update_graph()  # Update the graph to display in cm
        else:
            self.calibration_message.set("Calibration failed")


    def calibrate_close_calibration_window(self):
        if self.calibration_window:
            self.calibration_window.destroy()
            self.calibration_window = None



if __name__ == "__main__":
    root = tk.Tk()
    app = BreathingMonitorApp(root)
    root.mainloop()