# Bottle Detection using YOLOv8 + OpenCV + Matplotlib

### Specialized Bottle Detection System

A focused computer vision project that detects **bottles** in static images using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), **OpenCV**, and **Matplotlib**. This beginner-friendly prototype demonstrates object detection fundamentals with visual feedback through bounding boxes and checkmarks ✅.

##  Project Overview

This was my first hands-on project with YOLO and OpenCV, designed to learn:
- **Model Loading & Inference**: Using `yolov8m.pt` for detection
- **Class Filtering**: Specifically targeting bottles (COCO class ID 39)
- **Visual Annotations**: Drawing bounding boxes and checkmarks
- **Batch Processing**: Handling multiple images simultaneously
- **Results Visualization**: Creating organized matplotlib grids

##  Features

-  **Bottle-Specific Detection**: Filters for bottle objects only
-  **Bounding Box Visualization**: Clear object boundaries
-  **Center Checkmarks**: Green checkmarks on detected bottle centers
-  **Grid Display**: Organized matplotlib visualization
-  **Batch Processing**: Process multiple images at once
-  **Flexible Input**: Support for various image formats

##  Requirements

Create a `requirements.txt` file:

```
ultralytics>=8.0.0
opencv-python>=4.5.0
matplotlib>=3.3.0
numpy>=1.19.0
torch>=1.7.0
torchvision>=0.8.0
Pillow>=8.0.0
jupyter>=1.0.0
```

##  Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Iamm3taphorical/yolo-opencv-bottle-detection.git
   cd yolo-opencv-bottle-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model:**
   ```bash
   # The model will download automatically on first run
   # Or download manually:
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   ```

##  Usage

### Running the Jupyter Notebook

```bash
jupyter notebook Water_Bottle_Detection.ipynb
```

### Python Script Usage

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model = YOLO('yolov8m.pt')

# Define image paths
image_paths = [
    'path/to/image1.jpg',
    'path/to/image2.png',
    'path/to/image3.jpeg'
]

# Process images
for image_path in image_paths:
    results = model(image_path)
    
    # Filter for bottles (class 39)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            bottle_boxes = boxes[boxes.cls == 39]  # Filter for bottles
            # Process detections...
```

## Project Structure

```
yolo-opencv-bottle-detection/
├── Water_Bottle_Detection.ipynb    # Main Jupyter notebook
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── images/                         # Input images folder
├── output/                         # Detection results
├── models/                         # YOLO model files
└── examples/                       # Example images
```

##  What's Inside

### Core Functions:

- **`detect_bottles(...)`**: 
  - Handles model inference
  - Draws bounding boxes
  - Adds confidence scores
  - Filters bottle detections

- **`draw_checkmark(...)`**: 
  - Draws green checkmarks on detected bottle centers
  - Customizable size and color
  - Visual confirmation of detection

- **Batch Processing Pipeline**:
  - Processes multiple images
  - Creates matplotlib visualization grids
  - Saves annotated results

##  Visualization Features

### Bounding Boxes
- **Color**: Customizable (default: green for bottles)
- **Thickness**: Adjustable line width
- **Labels**: Class name + confidence score

### Checkmarks
- **Position**: Center of detected bottles
- **Color**: Green ✅
- **Size**: Proportional to bounding box

### Grid Layout
- **Organization**: Multiple images in subplot grid
- **Titles**: Image names and detection counts
- **Spacing**: Optimized for readability

##  Configuration

### Detection Parameters
- **Confidence Threshold**: Default 0.25 (adjustable)
- **Model Size**: YOLOv8m (medium) - balance of speed/accuracy
- **Target Class**: Bottle (COCO ID: 39)

### Visualisation Settings
- **Grid Size**: Auto-calculated based on image count
- **Figure Size**: Configurable for different screen sizes
- **Color Scheme**: Customizable for different preferences

##  Performance

### Model Specifications:
- **Model**: YOLOv8m (medium)
- **Target**: Bottle detection only
- **Speed**: ~50-100ms per image (GPU)
- **Accuracy**: High for clear bottle images

### Tested Scenarios:
- Water bottles
-  Glass bottles
- Plastic bottles
- Multiple bottles in scene
-  Various lighting conditions

##  Troubleshooting

### Common Issues:

1. **"No bottles detected"**
   - Lower confidence threshold
   - Ensure bottles are visible
   - Check image quality and lighting

2. **"Model not found"**
   ```bash
   # Download YOLOv8m model
   python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
   ```

3. **"Matplotlib display issues"**
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```

4. **"Memory errors"**
   - Process smaller image batches
   - Use CPU inference: `model = YOLO('yolov8m.pt', device='cpu')`

##  Results Examples

### Detection Output:
- **Bounding boxes** around detected bottles
- **Confidence scores** displayed on boxes
- **Green checkmarks** at bottle centers
- **Grid visualization** of all processed images
