获取QQ : 767172261



# -1
深度学习数据集 道路裂缝识别检测RDD2022数据集 道路病害检测数据集
使用YOLOv8训练**草原牛羊马目标检测数据集**，并基于训练好的牛羊马检测数据集的权重构建一个完整的检测系统
@[toc]


草原牛羊马目标检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/350f5d47a816455da89c3a1d236374f6.png)

3类别、2400张
YOLO、VOC格式

已经划分为训练集、验证集、测试集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b2844988682041aea4a4411d15c3a41b.png)


1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eff10547df604dca96b785cbb93d6372.png)

使用YOLOv8训练**草原牛羊马目标检测数据集**，并基于训练好的权重构建一个完整的检测系统（支持图片、视频或摄像头实时检测），我们将从以下几个方面进行详细讲解：

---

## ✅ 一、环境搭建

### 1. 安装CUDA驱动（可选）

确保你的显卡支持CUDA，并安装对应的NVIDIA驱动。如需GPU加速，请安装与PyTorch兼容的CUDA版本。

### 2. 安装Anaconda

从 [Anaconda官网](https://www.anaconda.com/products/distribution) 下载安装 Anaconda。

### 3. 创建虚拟环境

```bash
conda create -n grassland_detection python=3.9
conda activate grassland_detection
```

### 4. 安装依赖项

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python-headless
pip install PyQt5
```

---

## 📁 二、数据集准备与预处理

### 数据结构要求（YOLO格式）

请将数据组织为以下目录结构：

```
grassland_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每个图像文件（`.jpg`）对应一个 `.txt` 标注文件，每行表示一个目标：  
`class_id x_center y_center width height`（归一化坐标）

### 类别信息

- **类别数量**：3类
- **类别名称**：`['cattle', 'sheep', 'horse']`

---

## 🧾 三、创建 YOLO 配置文件 `data.yaml`

在项目根目录下创建 `data.yaml` 文件，内容如下：

```yaml
train: ./grassland_dataset/images/train
val: ./grassland_dataset/images/val
test: ./grassland_dataset/images/test

nc: 3
names: ['cattle', 'sheep', 'horse']
```

---

## 🚀 四、YOLOv8 模型训练

### 1. 使用预训练模型开始训练

```bash
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

参数说明：

- `model`: 使用的模型类型，如 `yolov8s.pt`（小模型）、`yolov8m.pt`（中）、`yolov8l.pt`（大）
- `data`: 指定数据配置文件
- `epochs`: 训练轮数
- `imgsz`: 输入图像尺寸
- `batch`: 批量大小（根据显存调整）

训练完成后，最佳权重保存路径为：`runs/detect/train/weights/best.pt`

---

## 🔍 五、模型推理与部署

### 单张图像预测

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 图像预测
results = model('path_to_your_image.jpg')
```

### 视频或摄像头预测

```python
results = model('path_to_video.mp4')  # 视频
results = model(0)  # 摄像头
```

---

## 📊 六、性能评估

使用验证集评估模型精度：

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

输出指标包括：

- mAP@0.5
- mAP@0.5:0.95
- 精确率（Precision）
- 召回率（Recall）

---

## 📦 七、模型导出

将模型导出为ONNX、TensorRT等格式以用于部署：

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

支持格式包括：`onnx`, `engine`, `tflite`, `pb`, `torchscript` 等。

---

## 💻 八、建立检测系统（GUI界面）

我们可以使用 PyQt5 构建一个简单的 GUI 应用程序来加载图像并显示检测结果。

### 示例代码：GUI 版本检测系统

```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("草原牛羊马检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.model = YOLO('runs/detect/train/weights/best.pt')  # 替换为你自己的best.pt路径

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel("上传图片", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        upload_btn = QPushButton("上传图片", self)
        upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(upload_btn)

        detect_btn = QPushButton("开始检测", self)
        detect_btn.clicked.connect(self.run_detection)
        layout.addWidget(detect_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image files (*.jpg *.png)")
        if fname:
            self.current_image_path = fname
            pixmap = QPixmap(fname).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def run_detection(self):
        results = self.model(self.current_image_path)
        image = cv2.imread(self.current_image_path)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{result.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
```

---

以上文字及代码仅供参考学习使用，。
