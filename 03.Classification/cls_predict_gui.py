import os
import glob
import shutil
import numpy as np
from PIL import Image, ImageQt
import cv2
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from datetime import datetime
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, 
                            QComboBox, QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem,
                            QGroupBox, QMessageBox, QHeaderView)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer  # 修复：添加QTimer导入


# 全局变量
CLASS_NAMES = None
IMAGE_SIZE = (320, 320)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class PredictThread(QThread):
    """预测线程，避免UI卡顿"""
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    update_table = pyqtSignal(list)
    predict_finished = pyqtSignal(str)

    def __init__(self, model_path, input_path, labels_path, save_root, recursive, file_action):
        super().__init__()
        self.model_path = model_path
        self.input_path = input_path
        self.labels_path = labels_path
        self.save_root = save_root
        self.recursive = recursive
        self.file_action = file_action
        self.model = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_root_with_timestamp = os.path.join(save_root, self.timestamp)

    def run(self):
        global CLASS_NAMES
        try:
            # 加载类别名称
            self.update_log.emit("加载类别名称...")
            CLASS_NAMES = self.load_class_names(self.labels_path)
            if not CLASS_NAMES:
                self.update_log.emit(f"错误：未找到类别文件 - {self.labels_path}")
                return

            # 加载模型
            self.update_log.emit("加载模型...")
            self.model = self.load_model(self.model_path)
            if not self.model:
                return

            # 收集图像路径
            self.update_log.emit("收集图像路径...")
            image_paths = self.collect_image_paths(self.input_path, self.recursive)
            if not image_paths:
                self.update_log.emit("未找到图像文件")
                return
            self.update_log.emit(f"共找到 {len(image_paths)} 张图像")

            # 初始化表格
            self.update_table.emit([["图像路径", "预测类别", "置信度(%)", "操作状态"]])

            # 预测并处理文件
            total = len(image_paths)
            for i, img_path in enumerate(image_paths):
                result = self.predict_single_image(self.model, img_path)
                status = "无"
                
                if result:
                    pred_class = result["pred_class"]
                    confidence = result["confidence"]
                    
                    if self.file_action in ['copy', 'move']:
                        self.handle_image_file(img_path, pred_class, self.save_root_with_timestamp, self.file_action)
                        status = "成功"
                    
                    self.update_table.emit([
                        [result["image_path"], pred_class, f"{confidence:.2f}", status]
                    ])
                else:
                    self.update_table.emit([[img_path, "预测失败", "-", "无"]])

                # 更新进度
                self.update_progress.emit(int((i+1)/total * 100))

            self.predict_finished.emit(f"预测完成！结果保存至：{self.save_root_with_timestamp if self.file_action != 'none' else '无'}")

        except Exception as e:
            self.update_log.emit(f"执行错误：{str(e)}")

    @staticmethod
    def load_class_names(labels_ini_path):
        if not os.path.exists(labels_ini_path):
            return None
        class_names = []
        with open(labels_ini_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_names.append(line.split('\t')[0] if '\t' in line else line)
        return class_names if class_names else None

    @staticmethod
    def preprocess_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMAGE_SIZE, Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)
        img_np = img_np.transpose(2, 0, 1)
        for i in range(3):
            img_np[i] = (img_np[i] / 255.0 - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
        return img_np[np.newaxis, ...]

    @staticmethod
    def load_model(model_path):
        try:
            ext = os.path.splitext(model_path)[1].lower()
            if ext == '.onnx':
                # 修复：仅使用可用的执行提供者
                providers = ['CPUExecutionProvider']  # 若没有CUDA则只用CPU
                # 检查是否有CUDA提供者
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                return ONNXModel(model_path, providers)
            elif ext in ['.trt', '.engine']:
                return TRTModel(model_path)
            else:
                raise ValueError(f"不支持的模型格式: {ext}")
        except Exception as e:
            QThread.currentThread().update_log.emit(f"模型加载失败：{str(e)}")
            return None

    @staticmethod
    def predict_single_image(model, image_path):
        try:
            img_tensor = PredictThread.preprocess_image(image_path)
            outputs = model.predict(img_tensor)
            probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
            pred_idx = np.argmax(probabilities, axis=1)[0]
            confidence = probabilities[0, pred_idx] * 100
            return {
                "image_path": image_path,
                "pred_class": CLASS_NAMES[pred_idx],
                "confidence": confidence
            }
        except Exception as e:
            QThread.currentThread().update_log.emit(f"预测失败 {image_path}：{str(e)}")
            return None

    @staticmethod
    def collect_image_paths(input_path, recursive=False):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
        image_paths = []
        
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            if any(ext in ext_list for ext_list in image_extensions):
                image_paths.append(input_path)
            else:
                QThread.currentThread().update_log.emit(f"警告：{input_path} 不是支持的图像格式")
            return image_paths
        
        elif os.path.isdir(input_path):
            for ext in image_extensions:
                pattern = os.path.join(input_path, '**', ext) if recursive else os.path.join(input_path, ext)
                image_paths.extend(glob.glob(pattern, recursive=recursive))
            return sorted(image_paths)
        
        else:
            QThread.currentThread().update_log.emit(f"错误：输入路径不存在 - {input_path}")
            return []

    @staticmethod
    def handle_image_file(image_path, pred_class, save_root, action):
        if action == 'none':
            return
        
        img_filename = os.path.basename(image_path)
        save_dir = os.path.join(save_root, pred_class)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, img_filename)
        
        count = 1
        while os.path.exists(save_path):
            name, ext = os.path.splitext(img_filename)
            save_path = os.path.join(save_dir, f"{name}_{count}{ext}")
            count += 1
        
        try:
            if action == 'copy':
                shutil.copy2(image_path, save_path)
                QThread.currentThread().update_log.emit(f"已复制: {image_path} -> {save_path}")
            elif action == 'move':
                shutil.move(image_path, save_path)
                QThread.currentThread().update_log.emit(f"已移动: {image_path} -> {save_path}")
        except Exception as e:
            QThread.currentThread().update_log.emit(f"文件操作失败 {image_path}：{str(e)}")


class ONNXModel:
    # 修复：接收providers参数
    def __init__(self, model_path, providers):
        self.session = ort.InferenceSession(
            model_path,
            providers=providers  # 使用传入的执行提供者
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        PredictThread.currentThread().update_log.emit(f"已加载ONNX模型: {model_path}")
        PredictThread.currentThread().update_log.emit(f"使用执行提供者: {providers}")

    def predict(self, img_tensor):
        outputs = self.session.run([self.output_name], {self.input_name: img_tensor})
        return outputs[0]


class TRTModel:
    def __init__(self, model_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        PredictThread.currentThread().update_log.emit(f"已加载TensorRT模型: {model_path}")

    def predict(self, img_tensor):
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(1, -1)


class ClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.predict_thread = None

    def init_ui(self):
        self.setWindowTitle("图像分类预测工具")
        self.setGeometry(100, 100, 1400, 800)

        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 1. 路径配置区域
        path_group = QGroupBox("路径配置")
        path_layout = QVBoxLayout()
        path_layout.setSpacing(8)
        
        # 模型路径
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型路径:"), 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\best_model.onnx")
        model_layout.addWidget(self.model_path_edit, 1)
        self.model_browse_btn = QPushButton("浏览")
        self.model_browse_btn.setFixedWidth(80)
        model_layout.addWidget(self.model_browse_btn, 0)
        path_layout.addLayout(model_layout)
        
        # 输入路径
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入路径:"), 0)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setText(r"D:\Min\Projects\VSCodeProjects\dataset\cls_test_data\data")
        input_layout.addWidget(self.input_path_edit, 1)
        self.input_browse_btn = QPushButton("浏览")
        self.input_browse_btn.setFixedWidth(80)
        input_layout.addWidget(self.input_browse_btn, 0)
        path_layout.addLayout(input_layout)
        
        # 类别文件路径
        labels_layout = QHBoxLayout()
        labels_layout.addWidget(QLabel("类别文件:"), 0)
        self.labels_path_edit = QLineEdit()
        self.labels_path_edit.setText(r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\classes.txt")
        labels_layout.addWidget(self.labels_path_edit, 1)
        self.labels_browse_btn = QPushButton("浏览")
        self.labels_browse_btn.setFixedWidth(80)
        labels_layout.addWidget(self.labels_browse_btn, 0)
        path_layout.addLayout(labels_layout)
        
        # 保存路径
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("保存根目录:"), 0)
        self.save_root_edit = QLineEdit()
        self.save_root_edit.setText(r"D:\Min\Projects\VSCodeProjects\dataset\cls_test_data\pred_result")
        save_layout.addWidget(self.save_root_edit, 1)
        self.save_browse_btn = QPushButton("浏览")
        self.save_browse_btn.setFixedWidth(80)
        save_layout.addWidget(self.save_browse_btn, 0)
        path_layout.addLayout(save_layout)
        
        path_group.setLayout(path_layout)
        main_layout.addWidget(path_group)

        # 2. 选项配置区域
        options_group = QGroupBox("操作选项")
        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(10, 10, 10, 10)
        options_layout.setSpacing(20)
        
        self.recursive_check = QCheckBox("递归遍历文件夹")
        self.recursive_check.setChecked(True)
        options_layout.addWidget(self.recursive_check)
        
        options_layout.addWidget(QLabel("文件操作:"))
        self.file_action_combo = QComboBox()
        self.file_action_combo.addItems(["copy", "move", "none"])
        self.file_action_combo.setCurrentText("copy")
        self.file_action_combo.setFixedWidth(100)
        options_layout.addWidget(self.file_action_combo)
        
        options_layout.addStretch()
        self.start_btn = QPushButton("开始预测")
        self.start_btn.setFixedWidth(120)
        self.start_btn.clicked.connect(self.start_prediction)
        options_layout.addWidget(self.start_btn)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # 3. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 4. 结果显示区域
        result_layout = QHBoxLayout()
        result_layout.setSpacing(10)
        
        # 表格显示预测结果
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["图像路径", "预测类别", "置信度(%)", "操作状态"])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        result_layout.addWidget(self.result_table, 3)
        
        # 日志显示
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        result_layout.addWidget(self.log_text, 1)
        
        main_layout.addLayout(result_layout, 5)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.onnx *.trt *.engine)")
        if path:
            self.model_path_edit.setText(path)

    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.input_path_edit.setText(path)

    def browse_labels(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择类别文件", "", "文本文件 (*.txt *.ini)")
        if path:
            self.labels_path_edit.setText(path)

    def browse_save(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if path:
            self.save_root_edit.setText(path)

    def start_prediction(self):
        # 检查路径是否有效
        model_path = self.model_path_edit.text()
        input_path = self.input_path_edit.text()
        labels_path = self.labels_path_edit.text()
        save_root = self.save_root_edit.text()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "路径错误", f"模型文件不存在: {model_path}")
            return
        if not os.path.exists(input_path):
            QMessageBox.warning(self, "路径错误", f"输入路径不存在: {input_path}")
            return
        if not os.path.exists(labels_path):
            QMessageBox.warning(self, "路径错误", f"类别文件不存在: {labels_path}")
            return
        if not os.path.exists(save_root) and self.file_action_combo.currentText() != "none":
            os.makedirs(save_root, exist_ok=True)

        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_table.setRowCount(0)
        self.log_text.clear()

        # 创建并启动预测线程
        self.predict_thread = PredictThread(
            model_path,
            input_path,
            labels_path,
            save_root,
            self.recursive_check.isChecked(),
            self.file_action_combo.currentText()
        )
        self.predict_thread.update_log.connect(self.update_log)
        self.predict_thread.update_progress.connect(self.update_progress)
        self.predict_thread.update_table.connect(self.update_table)
        self.predict_thread.predict_finished.connect(self.prediction_finished)
        self.predict_thread.start()

    def update_log(self, message):
        self.log_text.append(message)
        self.log_text.moveCursor(self.log_text.textCursor().End)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_table(self, data):
        for row_data in data:
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            for col, item_text in enumerate(row_data):
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if col == 0:
                    item.setToolTip(item_text)
                self.result_table.setItem(row, col, item)
        # 调整第一列宽度
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        # 恢复手动调整
        QTimer.singleShot(100, lambda: self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive))

    def prediction_finished(self, message):
        self.update_log(message)
        self.start_btn.setEnabled(True)
        QMessageBox.information(self, "完成", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClassificationApp()
    window.show()
    sys.exit(app.exec_())