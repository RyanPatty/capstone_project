# Spectrum Analyzer Tool - Software Design Document

## Team Members
Ryan O’Connor, Ayorinde Lawani, Richard Luthringshauser, Nathan Reed, Dinesh Sekar  
Capstone Course: SWE 7903, Fall 2023  
Professor: Dr. Reza Parizi

---

## 1. Introduction

### 1.1 Purpose
To provide a technical overview and rationale for the architecture, design, and implementation of the Spectrum Analyzer Tool.

### 1.2 Scope
A desktop Python tool that automates RF signal extraction from Spectrum Analyzer video footage using OpenCV, OCR, and PyQt5. Outputs structured CSV data to streamline analysis.

### 1.3 Goals
- Reduce manual labor and analysis time
- Improve accuracy and data integrity
- Provide an intuitive user interface

---

## 2. System Overview
Targeted for engineers at Robins AFB to automate RF signal detection. The tool reads MP4 videos, detects peaks in RF signals, and logs numeric results.

---

## 3. System Architecture

### Components
- **UI Module** (PyQt5): Handles user interaction.
- **Video Processor** (OpenCV): Detects RF signal peaks.
- **Data Storage**: Outputs CSV files.
- **Feedback Mechanism**: Shows progress bars and updates.
- **Integration Layer**: Coordinates between modules.

---

## 4. Detailed Design

### Key Classes
- `WelcomeDialog`: Initial UI interface.
- `VideoProcessorRunnable`: Multithreaded handler.
- `VideoProcessor`: Frame analysis and CSV output.
- `ProgressDialog`: Real-time progress.
- `App`: Application orchestration.

### Algorithms
- Detects signals frame-by-frame.
- Uses OpenCV for analysis and EasyOCR to extract spectrometer settings.

---

## 5. Data Design
All signal data is saved to CSV files:
- Fields: timestamp, frequency, power, min, max, average
- Format: `power_records_{filename}.csv`

---

## 6. Human Interface

### UI Flow
1. Welcome screen → video selection
2. Choose frame-skip interval
3. Processing screen (video, progress bar)
4. Export CSV results

### UX Focus
Designed for fast, intuitive use with minimal training.

---

## 7. Design Justification
- Modular architecture improves maintainability.
- Offline functionality required for secure lab environment.
- Tools selected for robustness: PyQt5, OpenCV, EasyOCR.
