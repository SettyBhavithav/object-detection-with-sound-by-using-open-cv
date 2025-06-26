# YOLO Model Checkpoint Weights

Place the pretrained YOLO model weights inside this directory:

- **`yolov3.weights`**: YOLOv3 pretrained weights ($236\text{ MB}$, trained on COCO dataset).

## 📥 How to Download `yolov3.weights`

Download the official weight file directly from PJReddie or HuggingFace:

- **Official Direct URL**: [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

### Command Line Download Options:

#### Using PowerShell:
```powershell
Invoke-WebRequest -Uri "https://pjreddie.com/media/files/yolov3.weights" -OutFile "models/yolov3.weights"
```

#### Using `curl` / `wget`:
```bash
curl -o models/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
```

> **Note**: Heavy binary weight files (`.weights`) are ignored by `.gitignore` to comply with GitHub file size guidelines. If no weight file is present, the system automatically runs in **Synthetic Demo Mode** for zero-error verification.
