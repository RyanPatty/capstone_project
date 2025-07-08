# 📡 Spectrum Analyzer Tool

An offline desktop application built in Python that **automates the extraction of RF signal data** from long-form spectrum analyzer videos. Designed for **RF engineers and analysts**, this tool reduces hours of manual video review to just minutes—**no internet required**.

> Built for Robins AFB. Developed as part of a graduate capstone at Kennesaw State University.

---

## Key Features

✅ **Multi-Video Processing** — Load and batch-analyze multiple `.mp4` files  
✅ **Frame Skipping** — Adjust how frequently frames are sampled (1s, 5s, 10s, etc.)  
✅ **Peak Detection** — Automatically identifies RF signal peaks using OpenCV  
✅ **OCR Integration** — Uses EasyOCR to detect and extract spectrometer settings  
✅ **CSV Export** — Saves clean, structured output: `timestamp`, `frequency`, `power`, `min`, `max`, `average`  
✅ **Real-Time Feedback** — Progress bars and elapsed time displays  
✅ **Offline-First** — 100% local processing; no cloud or web dependencies  
✅ **Built with PyQt5** — Intuitive, responsive desktop UI

---

## 🛠️ Setup & Installation

### Option 1: Dev Setup (Python 3.11+)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

### Option 2: End-User (Windows Only)
- Unzip the `.zip` package (~1.3 GB)
- Double-click `main.exe`
- Select your videos, choose frame-skip, and process

---

## 🧱 System Architecture

```
┌────────────────────┐
│   User Interface   │ ◄──── User selects videos & parameters
└────────────────────┘
          │
          ▼
┌────────────────────┐
│  Integration Layer │ ◄──── Coordinates video input/output
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ Video Processing   │ ◄──── Frame analysis via OpenCV + OCR
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ Data Storage (CSV) │ ◄──── Clean, numeric export
└────────────────────┘
```

---

## 📂 Example Output (CSV Fields)

```csv
timestamp, frequency, power, min, max, average
12:34:56, 2.45GHz, -60dBm, -62, -58, -60
```

---

## 👨‍💻 Team

Developed by:  
Ryan O’Connor, Richard Lutheringhauser, Nathan Reed, Dinesh Sekar, Ayorinde Lawani  
🎓 Capstone Project – KSU MS in Software Engineering – Fall 2023

---

## 🧪 Status

✅ Final Version 1.0 — Delivered December 2023  
📦 Includes user and developer documentation  
🔒 Designed for secure, air-gapped environments

---

## 📞 Contact

Questions or reuse inquiries?  
📧 Ryan O'Connor: [ryan@darkoxygensoftware.com](mailto:ryan@darkoxygensoftware.com)
