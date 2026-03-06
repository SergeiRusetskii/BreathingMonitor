# Breathing Monitor

A real-time breathing pattern monitoring application that uses computer vision to track respiratory movements through video analysis.

## Features

### Core Functionality
- **Real-time Video Analysis**: Processes video streams from cameras or RTSP sources to detect breathing patterns
- **Live Graphical Display**: Shows breathing amplitude over time with configurable time windows
- **Patient Management**: Load and manage patient data from CSV/Excel files
- **Data Export**: Save breathing data to CSV files for further analysis
- **Multi-threading Support** (v4.0+): Optimized performance with separate capture and processing threads

### Advanced Features
- **Calibration System**: Automatic and manual calibration for accurate measurements
- **Customizable Thresholds**: Configure upper and lower breathing rate limits
- **Region of Interest Selection**: Define specific areas for monitoring
- **Distance Correction**: Compensate for camera distance variations
- **Settings Persistence**: Save and load application configurations

## Version History

This repository uses Git tags for version management. You can view all versions using:

```bash
git tag
```

To checkout a specific version:
```bash
git checkout v4.0  # Latest stable version
git checkout v3.9  # Previous version
```

**Major Versions:**
- **v1.0**: Initial basic breathing monitor
- **v2.0**: Enhanced features and improved UI
- **v3.x**: Iterative improvements (v3.0 through v3.9)
- **v4.0**: Major release with multi-threading support and improved stability

View the complete changelog with:
```bash
git log --oneline --decorate
```

## Requirements

```
python >= 3.7
tkinter
opencv-python (cv2)
numpy
pandas
matplotlib
```

## Project Structure

- `breathing_monitor.py`: Tkinter interface and application logic
- `monitor_threads.py`: Worker threads for video capture and processing
- `settings*.txt`: Example configuration files

## Configuration

The application uses a `settings.txt` file for configuration:

- `sample_rate`: Data sampling frequency (Hz)
- `graph_window_sec`: Time window for graph display (seconds)
- `capture_top/bottom/left/right`: Region of interest coordinates
- `calibration_factor`: Calibration multiplier for measurements
- `lower_threshold/upper_threshold`: Breathing rate limits
- `camera_url`: Video source (file path or RTSP URL)

## Usage

1. **Camera Setup**: Configure your camera source in settings.txt
2. **Patient Selection**: Load patient data from your own CSV or Excel files
3. **Region Selection**: Define the monitoring area on the video feed
4. **Calibration**: Run calibration to ensure accurate measurements
5. **Monitoring**: Start real-time breathing pattern analysis
6. **Data Export**: Save collected data for analysis

## Patient Database

The application supports loading patient records from external CSV or Excel files.

## Technical Details

### Architecture (v4.0)
- **SharedData Class**: Thread-safe data container
- **VideoCaptureThread**: Dedicated video capture handling
- **VideoProcessThread**: Frame processing and analysis
- **BreathingMonitorApp**: Main GUI and coordination

### Signal Processing
- Computer vision-based motion detection
- Amplitude extraction from video frames
- Real-time data filtering and smoothing
- Configurable sensitivity thresholds

## Medical Disclaimer

This application is intended for research and educational purposes. It should not be used as a substitute for professional medical equipment or diagnosis. Always consult healthcare professionals for medical monitoring needs.

## License

This project is released under the Breathing Monitor Community License. See `LICENSE`.
