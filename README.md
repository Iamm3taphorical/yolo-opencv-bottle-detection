# YOLOv8 + OpenCV + Matplotlib | Bottle Detection 

### First project using YOLO and Opencv 

This project detects **bottles** in static images using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), **OpenCV**, and **Matplotlib**. I tested this on a set of 12 screenshots, and the detections are visualised with bounding boxes and checkmarks ✅.

It's a beginner-level prototype that helped me get my hands dirty with:
- Model loading & inference using `yolov8m.pt`
- Class filtering for bottles (class ID 39)
- Drawing bounding boxes and checkmarks on detected objects
- Visualising output using `matplotlib`

---

## 📁 What’s Inside

- `detect_bottles(...)`: handles model inference + drawing boxes + adding checkmarks
- `draw_checkmark(...)`: draws a green checkmark on detected bottle centres
- Batch image processing with `matplotlib` visualisation grid

---

## 🔧 How to Run

1. Install dependencies:

```bash
pip install ultralytics opencv-python matplotlib torch numpy
```

2. Replace image_paths with paths to your images

3. Run the script or Colab notebook
