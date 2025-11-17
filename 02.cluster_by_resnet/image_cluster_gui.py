import os
import sys
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from glob import glob
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QLineEdit,
                               QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                               QProgressBar, QMessageBox, QGroupBox, QVBoxLayout, QHBoxLayout,
                               QWidget, QFrame, QTextEdit, QSplitter)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QTextCursor


# ------------------------------
# 核心功能：图像读写与特征聚类
# ------------------------------
def cv2_imread(img_path):
    try:
        with open(img_path, 'rb') as f:
            img_data = f.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return f"读取失败：{str(e)}"

def cv2_imwrite(save_path, img):
    try:
        ext = os.path.splitext(save_path)[1].lower() or '.jpg'
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext in ['.jpg', '.jpeg'] else \
                      [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        result, buf = cv2.imencode(ext, img, encode_param)
        if result:
            with open(save_path, 'wb') as f:
                f.write(buf)
            return True
        return False
    except Exception as e:
        return f"保存失败：{str(e)}"

class FeatureExtractor:
    def __init__(self, model_name='resnet18', progress_callback=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name.lower()
        self.progress_callback = progress_callback  # 模型加载进度回调

    def load_model(self):
        """加载模型并报告进度（修复torchvision的weights参数警告）"""
        if self.progress_callback:
            self.progress_callback(10, "初始化特征提取器...")
        
        try:
            # 修复：使用weights参数替代deprecated的pretrained
            if self.model_name == 'mobilenet':
                if self.progress_callback:
                    self.progress_callback(30, "加载 MobileNet 模型...")
                self.model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
                self.model = torch.nn.Sequential(*list(self.model.features) + [torch.nn.AdaptiveAvgPool2d((1, 1))])
                self.feature_dim = 1280

            elif self.model_name == 'resnet34':
                if self.progress_callback:
                    self.progress_callback(30, "加载 ResNet34 模型...")
                self.model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
                self.feature_dim = 512

            elif self.model_name == 'resnet50':
                if self.progress_callback:
                    self.progress_callback(30, "加载 ResNet50 模型...")
                self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
                self.feature_dim = 2048

            elif self.model_name == 'resnet18':
                if self.progress_callback:
                    self.progress_callback(30, "加载 ResNet18 模型...")
                self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
                self.feature_dim = 512

            else:
                raise ValueError(f"不支持的模型：{self.model_name}")

            if self.progress_callback:
                self.progress_callback(70, "模型加载完成，转移到设备...")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            if self.progress_callback:
                self.progress_callback(100, "特征提取器准备就绪")
            return True

        except Exception as e:
            if self.progress_callback:
                self.progress_callback(-1, f"模型加载失败：{str(e)}")
            return False

    def extract(self, img_path):
        try:
            img = cv2_imread(img_path)
            if isinstance(img, str):
                return img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(img_tensor)
            return feature.squeeze().cpu().numpy()
        except Exception as e:
            return f"提取失败：{str(e)}"

class ImageCluster:
    def __init__(self, extractor, cluster_method='kmeans', n_clusters=5, eps=0.5, min_samples=5):
        self.extractor = extractor
        self.cluster_method = cluster_method
        self.features = []
        self.image_paths = []
        self.model = KMeans(n_clusters=n_clusters, random_state=42) if cluster_method == 'kmeans' else \
                      DBSCAN(eps=eps, min_samples=min_samples)

    def load_images(self, folder_path, recursive=False, progress_callback=None):
        if not os.path.isdir(folder_path):
            return f"文件夹不存在：{folder_path}"
        
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
        img_paths = []
        for ext in img_extensions:
            if recursive:
                img_paths.extend(glob(os.path.join(folder_path, '**', ext), recursive=True))
            else:
                img_paths.extend(glob(os.path.join(folder_path, ext)))
        
        if not img_paths:
            return "未找到任何图像文件"
        
        total = len(img_paths)
        for i, img_path in enumerate(img_paths):
            feature = self.extractor.extract(img_path)
            if not isinstance(feature, str):
                self.features.append(feature)
                self.image_paths.append(img_path)
            if progress_callback:
                progress_callback(int((i+1)/total * 100))
        
        return f"成功提取 {len(self.features)}/{total} 张图像特征"

    def cluster(self, save_dir, progress_callback=None):
        if len(self.features) < 2:
            return "特征数量不足，无法聚类"
        
        if progress_callback:
            progress_callback(50)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        labels = self.model.fit_predict(features_scaled)
        if progress_callback:
            progress_callback(70)
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_dir = os.path.join(save_dir, f"cluster_{label}")
            os.makedirs(label_dir, exist_ok=True)
            
            for i, img_path in enumerate(self.image_paths):
                if labels[i] == label:
                    img = cv2_imread(img_path)
                    if not isinstance(img, str):
                        img_name = os.path.basename(img_path)
                        save_path = os.path.join(label_dir, img_name)
                        counter = 1
                        while os.path.exists(save_path):
                            name, ext = os.path.splitext(img_name)
                            save_path = os.path.join(label_dir, f"{name}_{counter}{ext}")
                            counter += 1
                        cv2_imwrite(save_path, img)
        
        if progress_callback:
            progress_callback(100)
        return f"聚类完成，共 {len(unique_labels)} 个类别，结果保存至：{save_dir}"


# ------------------------------
# 后台线程：修复信号传递问题
# ------------------------------
class ClusterThread(QThread):
    progress_updated = Signal(int)  # 整体进度 (0-100)
    status_updated = Signal(str)    # 状态文本
    console_updated = Signal(str)   # 控制台输出
    model_progress_updated = Signal(int, str)  # 模型加载进度 (值, 文本)
    result_ready = Signal(str)  # 新增：用于传递结果的信号（替代finished）

    def __init__(self, source_folder, save_folder, model_name, cluster_method, 
                 n_clusters, eps, min_samples, recursive):
        super().__init__()
        self.source_folder = source_folder
        self.save_folder = save_folder
        self.model_name = model_name
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.recursive = recursive

    def run(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.save_folder, f"{self.model_name}_clusters_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            self.console_updated.emit(f"创建保存目录: {save_dir}")

            # 初始化特征提取器并显示加载进度
            self.status_updated.emit(f"正在加载 {self.model_name} 模型...")
            extractor = FeatureExtractor(
                model_name=self.model_name,
                progress_callback=lambda val, msg: self.model_progress_updated.emit(val, msg)
            )
            
            # 加载模型（带进度反馈）
            if not extractor.load_model():
                self.result_ready.emit("模型加载失败，无法继续")  # 修复：使用新信号
                return

            self.console_updated.emit(f"使用计算设备: {extractor.device}")
            self.status_updated.emit("模型加载完成，准备提取特征")

            # 初始化聚类器
            cluster = ImageCluster(
                extractor,
                cluster_method=self.cluster_method,
                n_clusters=self.n_clusters,
                eps=self.eps,
                min_samples=self.min_samples
            )

            # 加载图像并提取特征（占总进度的40%）
            self.status_updated.emit("加载图像并提取特征...")
            self.console_updated.emit("开始扫描图像文件...")
            load_result = cluster.load_images(
                self.source_folder,
                recursive=self.recursive,
                progress_callback=lambda v: self.progress_updated.emit(40 * v // 100)  # 0-40%
            )
            self.console_updated.emit(load_result)
            self.status_updated.emit(load_result)

            if "成功提取" in load_result:
                # 执行聚类（占总进度的60%）
                self.status_updated.emit("执行聚类算法...")
                self.console_updated.emit(f"开始{self.cluster_method}聚类...")
                cluster_result = cluster.cluster(
                    save_dir,
                    progress_callback=lambda v: self.progress_updated.emit(40 + 60 * v // 100)  # 40-100%
                )
                self.console_updated.emit(cluster_result)
                self.result_ready.emit(cluster_result)  # 修复：使用新信号
            else:
                self.result_ready.emit(f"警告：{load_result}")  # 修复：使用新信号

        except Exception as e:
            err_msg = f"错误：{str(e)}"
            self.console_updated.emit(err_msg)
            self.result_ready.emit(err_msg)  # 修复：使用新信号


# ------------------------------
# 主界面：PySide6实现
# ------------------------------
class ClusterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像聚类工具")
        self.setGeometry(100, 100, 900, 700)
        self.init_ui()

    def init_ui(self):
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 顶部控制区域与底部控制台的分割器
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # 1. 顶部控制区域
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(15)
        splitter.addWidget(control_widget)

        # 1.1 文件夹选择区域
        folder_group = QGroupBox("文件夹设置")
        folder_layout = QVBoxLayout()
        folder_layout.setSpacing(10)

        # 源文件夹
        src_layout = QHBoxLayout()
        self.src_edit = QLineEdit()
        self.src_btn = QPushButton("浏览...")
        self.src_btn.clicked.connect(self.choose_source)
        src_layout.addWidget(QLabel("图像文件夹："))
        src_layout.addWidget(self.src_edit, 1)
        src_layout.addWidget(self.src_btn)
        folder_layout.addLayout(src_layout)

        # 保存文件夹
        save_layout = QHBoxLayout()
        self.save_edit = QLineEdit()
        self.save_btn = QPushButton("浏览...")
        self.save_btn.clicked.connect(self.choose_save)
        save_layout.addWidget(QLabel("保存目录："))
        save_layout.addWidget(self.save_edit, 1)
        save_layout.addWidget(self.save_btn)
        folder_layout.addLayout(save_layout)

        folder_group.setLayout(folder_layout)
        control_layout.addWidget(folder_group)

        # 1.2 参数设置区域
        param_group = QGroupBox("参数设置")
        param_layout = QHBoxLayout()
        left_param = QVBoxLayout()
        right_param = QVBoxLayout()
        param_layout.addLayout(left_param, 1)
        param_layout.addLayout(right_param, 1)

        # 左侧参数：模型和聚类方法
        left_param.addWidget(QLabel("特征提取模型："))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["resnet18", "resnet34", "resnet50", "mobilenet"])
        left_param.addWidget(self.model_combo)

        left_param.addSpacing(15)
        left_param.addWidget(QLabel("聚类算法："))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["kmeans", "dbscan"])
        self.method_combo.currentTextChanged.connect(self.update_param_visibility)
        left_param.addWidget(self.method_combo)

        left_param.addStretch()

        # 右侧参数：KMeans
        self.kmeans_frame = QFrame()
        kmeans_layout = QVBoxLayout(self.kmeans_frame)
        kmeans_layout.addWidget(QLabel("聚类类别数："))
        self.n_clusters = QSpinBox()
        self.n_clusters.setRange(2, 100)
        self.n_clusters.setValue(8)
        kmeans_layout.addWidget(self.n_clusters)

        # 右侧参数：DBSCAN（默认隐藏）
        self.dbscan_frame = QFrame()
        dbscan_layout = QVBoxLayout(self.dbscan_frame)
        dbscan_layout.addWidget(QLabel("eps（密度半径）："))
        self.eps = QDoubleSpinBox()
        self.eps.setRange(0.1, 5.0)
        self.eps.setValue(0.5)
        self.eps.setSingleStep(0.1)
        dbscan_layout.addWidget(self.eps)

        dbscan_layout.addWidget(QLabel("最小样本数："))
        self.min_samples = QSpinBox()
        self.min_samples.setRange(2, 50)
        self.min_samples.setValue(5)
        dbscan_layout.addWidget(self.min_samples)

        # 递归选项
        self.recursive_check = QCheckBox("递归读取子文件夹")
        self.recursive_check.setChecked(True)
        right_param.addWidget(self.recursive_check)

        right_param.addWidget(self.kmeans_frame)
        self.dbscan_frame.hide()  # 初始隐藏DBSCAN参数

        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)

        # 1.3 进度和状态区域
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.status_label)
        control_layout.addLayout(progress_layout)

        # 1.4 运行按钮
        self.run_btn = QPushButton("开始聚类")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self.start_clustering)
        control_layout.addWidget(self.run_btn)

        control_layout.addStretch()

        # 2. 底部控制台区域
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(5, 5, 5, 5)
        
        console_label = QLabel("控制台输出：")
        console_label.setFont(QFont("SimHei", 9))
        console_layout.addWidget(console_label)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 10))
        self.console.setStyleSheet("background-color: #f5f5f5;")
        console_layout.addWidget(self.console)
        
        splitter.addWidget(console_widget)
        
        # 设置分割器初始比例（控制区:控制台 = 3:2）
        splitter.setSizes([420, 280])

        # 模型加载进度条（默认隐藏）
        self.model_progress_bar = QProgressBar()
        self.model_progress_bar.setRange(0, 100)
        self.model_progress_bar.setValue(0)
        self.model_progress_bar.setHidden(True)
        control_layout.insertWidget(3, self.model_progress_bar)  # 插入到进度条上方

    def update_param_visibility(self, method):
        if method == "kmeans":
            self.dbscan_frame.hide()
            self.kmeans_frame.show()
        else:
            self.kmeans_frame.hide()
            self.dbscan_frame.show()

    def choose_source(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self.src_edit.setText(folder)
            self.console_updated(f"选择图像文件夹: {folder}")

    def choose_save(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if folder:
            self.save_edit.setText(folder)
            self.console_updated(f"选择保存目录: {folder}")

    def console_updated(self, text):
        """向控制台添加文本并自动滚动到底部"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{timestamp}] {text}")
        self.console.moveCursor(QTextCursor.End)

    def start_clustering(self):
        source_folder = self.src_edit.text().strip()
        save_folder = self.save_edit.text().strip()

        if not source_folder or not os.path.isdir(source_folder):
            QMessageBox.critical(self, "错误", "请选择有效的图像文件夹")
            return
        if not save_folder:
            QMessageBox.critical(self, "错误", "请选择保存目录")
            return

        # 重置状态
        self.console.clear()
        self.progress_bar.setValue(0)
        self.model_progress_bar.setValue(0)
        self.model_progress_bar.setHidden(False)  # 显示模型加载进度条

        # 禁用按钮
        self.run_btn.setEnabled(False)
        self.run_btn.setText("运行中...")
        self.status_label.setText("准备开始...")

        # 启动后台线程
        self.cluster_thread = ClusterThread(
            source_folder=source_folder,
            save_folder=save_folder,
            model_name=self.model_combo.currentText(),
            cluster_method=self.method_combo.currentText(),
            n_clusters=self.n_clusters.value(),
            eps=self.eps.value(),
            min_samples=self.min_samples.value(),
            recursive=self.recursive_check.isChecked()
        )
        self.cluster_thread.progress_updated.connect(self.progress_bar.setValue)
        self.cluster_thread.status_updated.connect(self.status_label.setText)
        self.cluster_thread.console_updated.connect(self.console_updated)
        self.cluster_thread.model_progress_updated.connect(self.update_model_progress)
        self.cluster_thread.result_ready.connect(self.on_clustering_finished)  # 修复：连接新信号
        self.cluster_thread.start()

    def update_model_progress(self, value, text):
        """更新模型加载进度"""
        if value == -1:  # 错误状态
            self.model_progress_bar.setStyleSheet("QProgressBar { color: red; }")
            self.status_label.setText(text)
            self.console_updated(text)
        else:
            self.model_progress_bar.setValue(value)
            self.status_label.setText(text)
            self.console_updated(text)

    def on_clustering_finished(self, result):
        """聚类完成后的处理（修复参数传递）"""
        self.model_progress_bar.setHidden(True)  # 隐藏模型加载进度条
        self.status_label.setText(result)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始聚类")
        
        if "错误" in result:
            QMessageBox.critical(self, "执行失败", result)
        elif "警告" in result:
            QMessageBox.warning(self, "警告", result)
        else:
            QMessageBox.information(self, "成功", result)


# ------------------------------
# 主程序入口
# ------------------------------
if __name__ == "__main__":
    # 确保中文显示正常
    import matplotlib
    matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 过滤torchvision的弃用警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

    app = QApplication(sys.argv)
    window = ClusterGUI()
    window.show()
    sys.exit(app.exec())