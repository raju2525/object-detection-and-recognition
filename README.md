# 🧠 Integrated Recognition & Detection using Convolutional Networks (YOLOv5 + PyQt5)

An **AI-powered desktop application** that integrates **object recognition and detection** using **Convolutional Neural Networks** (YOLOv5).  
Built with **PyTorch**, **OpenCV**, and a sleek **PyQt5 GUI**, it supports **image, video, and real-time webcam inference**.

---

## 🚀 Features

- 🖼️ **Upload Image** – Detect objects in any uploaded image.  
- 🎥 **Upload Video** – Run detection across full video files.  
- 📷 **Webcam Feed** – Real-time detection directly from your webcam.  
- 🧩 **Integrated GUI** – Clean PyQt5 interface with easy navigation.  
- ⚡ **Real-Time Inference** – Optimized YOLOv5 model for smooth performance.  
- 🧰 **Custom Model Ready** – Plug in your own trained `best.pt` weights instantly.  

---

## 🧪 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.x |
| **Frameworks** | PyTorch, OpenCV, PyQt5 |
| **Model** | YOLOv5 (Ultralytics) |
| **Interface** | Desktop GUI |
| **Detection Tasks** | Object recognition and localization |

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Integrated-Recognition-Detection.git
cd Integrated-Recognition-Detection

```
---

## 🖼️ GUI Preview
![download](https://github.com/user-attachments/assets/03160430-8515-429f-83e1-58136e05127b)
---

## ⚙️ Using Your Own Model
- If you’ve trained your own YOLOv5 model (e.g., for road signs, grains, or gestures):
- edit this line in yolo_gui.py:
- ``` model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) ```
- ⬇️ Change it to:
- ``` model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) ```
- Now it will use your custom-trained model for detection.

---





