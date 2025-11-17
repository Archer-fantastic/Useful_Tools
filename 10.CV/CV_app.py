import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QFileDialog, 
                            QMessageBox, QGroupBox, QTabWidget, QGridLayout, QSpinBox,
                            QRadioButton, QButtonGroup, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None  # 原始图像(BGR格式)
        self.processed_image = None  # 处理后图像(BGR格式)
        self.image_path = ""  # 图像路径
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        # 窗口基本设置
        self.setWindowTitle("图像变换工具")
        self.setGeometry(100, 100, 1250, 800)
        self.setMinimumSize(1100, 700)
        
        # 设置全局字体，确保中文显示清晰
        font = QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.setFont(font)
        
        # 主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)
        
        # 左侧控制面板（采用滚动区域，避免空间不足）
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(350)  # 增加宽度，确保内容不拥挤
        control_scroll.setMaximumWidth(400)
        control_scroll.setStyleSheet("QScrollArea {border: none;}")
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("图像变换工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            margin: 5px 0 15px 0; 
            padding: 8px;
            background-color: #f0f5f9;
            border-radius: 5px;
        """)
        control_layout.addWidget(title_label)
        
        # 加载图像区域
        load_group = QGroupBox("图像加载")
        load_layout = QVBoxLayout()
        load_layout.setContentsMargins(12, 12, 12, 12)
        load_layout.setSpacing(10)
        
        self.load_btn = QPushButton("选择图像")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("""
            padding: 8px; 
            margin: 3px 0;
            background-color: #e3f2fd;
            border: 1px solid #bbdefb;
            font-size: 10px;
        """)
        self.load_btn.setMinimumHeight(35)
        
        self.image_info = QLabel("未加载图像")
        self.image_info.setAlignment(Qt.AlignCenter)
        self.image_info.setStyleSheet("""
            color: #546e7a; 
            font-size: 10px; 
            margin: 3px 0;
            padding: 8px;
            background-color: #f9fbe7;
            border-radius: 3px;
            min-height: 50px;
            text-align: center;
        """)
        
        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.image_info)
        load_group.setLayout(load_layout)
        control_layout.addWidget(load_group)
        
        # 操作选择区域
        operation_group = QGroupBox("处理操作")
        operation_layout = QVBoxLayout()
        operation_layout.setContentsMargins(12, 12, 12, 12)
        
        self.operation_combo = QComboBox()
        self.operations = [
            "原始图像", "灰度图转换", "均值滤波", "中值滤波",
            "Sobel边缘检测", "Canny边缘检测", "01阈值分割", 
            "图像反转", "腐蚀操作", "膨胀操作"
        ]
        self.operation_combo.addItems(self.operations)
        self.operation_combo.currentIndexChanged.connect(self.on_operation_changed)
        self.operation_combo.setStyleSheet("""
            padding: 6px; 
            margin: 3px 0;
            background-color: white;
            border: 1px solid #bdbdbd;
            font-size: 10px;
        """)
        self.operation_combo.setMinimumHeight(30)
        
        operation_layout.addWidget(self.operation_combo)
        operation_group.setLayout(operation_layout)
        control_layout.addWidget(operation_group)
        
        # 参数控制区域（重新组织标签页，避免过多标签）
        self.param_tab = QTabWidget()
        # 优化标签样式，增加宽度和间距
        self.param_tab.setStyleSheet("""
            QTabBar::tab {
                height: 32px; 
                min-width: 90px;  /* 增加标签宽度 */
                margin-right: 2px;  /* 增加标签间距 */
                padding: 0 12px;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background-color: #e3f2fd;
                border-bottom: 2px solid #2196f3;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        
        # 合并部分标签页，减少标签数量
        self.basic_params = QWidget()  # 基础参数（合并通用+滤波）
        basic_layout = QVBoxLayout(self.basic_params)
        basic_layout.setContentsMargins(10, 10, 10, 10)
        basic_layout.setSpacing(12)
        
        # 阈值分割参数（移至基础参数页）
        threshold_group = QGroupBox("01阈值分割参数")
        threshold_layout = QVBoxLayout()
        threshold_layout.setContentsMargins(8, 8, 8, 8)
        threshold_layout.setSpacing(8)
        
        # 阈值类型选择
        self.threshold_type_group = QButtonGroup()
        threshold_type_layout = QHBoxLayout()
        threshold_type_layout.setSpacing(10)
        
        self.thresh_binary = QRadioButton("二值化")
        self.thresh_binary_inv = QRadioButton("反二值化")
        self.thresh_binary.setChecked(True)
        
        self.threshold_type_group.addButton(self.thresh_binary, 0)
        self.threshold_type_group.addButton(self.thresh_binary_inv, 1)
        
        threshold_type_layout.addWidget(self.thresh_binary)
        threshold_type_layout.addWidget(self.thresh_binary_inv)
        threshold_layout.addLayout(threshold_type_layout)
        
        # 阈值滑块
        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(8)
        self.threshold_label = QLabel("阈值: 128")
        self.threshold_label.setFixedWidth(70)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        threshold_row.addWidget(self.threshold_label)
        threshold_row.addWidget(self.threshold_slider)
        threshold_layout.addLayout(threshold_row)
        threshold_group.setLayout(threshold_layout)
        basic_layout.addWidget(threshold_group)
        
        # 均值滤波参数（移至基础参数页）
        blur_group = QGroupBox("均值滤波参数")
        blur_layout = QVBoxLayout()
        blur_layout.setContentsMargins(8, 8, 8, 8)
        
        blur_row = QHBoxLayout()
        blur_row.setSpacing(10)
        blur_row.addWidget(QLabel("核大小:"))
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(3, 11)
        self.blur_kernel.setSingleStep(2)
        self.blur_kernel.setValue(3)
        self.blur_kernel.valueChanged.connect(self.on_param_changed)
        blur_row.addWidget(self.blur_kernel)
        blur_layout.addLayout(blur_row)
        blur_group.setLayout(blur_layout)
        basic_layout.addWidget(blur_group)
        
        # 中值滤波参数（移至基础参数页）
        median_group = QGroupBox("中值滤波参数")
        median_layout = QVBoxLayout()
        median_layout.setContentsMargins(8, 8, 8, 8)
        
        median_row = QHBoxLayout()
        median_row.setSpacing(10)
        median_row.addWidget(QLabel("核大小:"))
        self.median_kernel = QSpinBox()
        self.median_kernel.setRange(3, 9)
        self.median_kernel.setSingleStep(2)
        self.median_kernel.setValue(3)
        self.median_kernel.valueChanged.connect(self.on_param_changed)
        median_row.addWidget(self.median_kernel)
        median_layout.addLayout(median_row)
        median_group.setLayout(median_layout)
        basic_layout.addWidget(median_group)
        
        basic_layout.addStretch()
        self.param_tab.addTab(self.basic_params, "基础参数")
        
        # 边缘检测参数页（保持不变）
        self.edge_params = QWidget()
        edge_layout = QVBoxLayout(self.edge_params)
        edge_layout.setContentsMargins(10, 10, 10, 10)
        edge_layout.setSpacing(12)
        
        # Sobel参数
        sobel_group = QGroupBox("Sobel参数")
        sobel_layout = QVBoxLayout()
        sobel_layout.setContentsMargins(8, 8, 8, 8)
        
        sobel_row = QHBoxLayout()
        sobel_row.setSpacing(10)
        sobel_row.addWidget(QLabel("核大小:"))
        self.sobel_kernel = QSpinBox()
        self.sobel_kernel.setRange(3, 7)
        self.sobel_kernel.setSingleStep(2)
        self.sobel_kernel.setValue(3)
        self.sobel_kernel.valueChanged.connect(self.on_param_changed)
        sobel_row.addWidget(self.sobel_kernel)
        sobel_layout.addLayout(sobel_row)
        sobel_group.setLayout(sobel_layout)
        edge_layout.addWidget(sobel_group)
        
        # Canny参数
        canny_group = QGroupBox("Canny参数")
        canny_layout = QVBoxLayout()
        canny_layout.setContentsMargins(8, 8, 8, 8)
        canny_layout.setSpacing(8)
        
        canny_row1 = QHBoxLayout()
        canny_row1.setSpacing(8)
        self.canny_label1 = QLabel("阈值1: 100")
        self.canny_label1.setFixedWidth(70)
        self.canny_slider1 = QSlider(Qt.Horizontal)
        self.canny_slider1.setRange(0, 255)
        self.canny_slider1.setValue(100)
        self.canny_slider1.valueChanged.connect(lambda v: self.update_canny(1, v))
        
        canny_row1.addWidget(self.canny_label1)
        canny_row1.addWidget(self.canny_slider1)
        
        canny_row2 = QHBoxLayout()
        canny_row2.setSpacing(8)
        self.canny_label2 = QLabel("阈值2: 200")
        self.canny_label2.setFixedWidth(70)
        self.canny_slider2 = QSlider(Qt.Horizontal)
        self.canny_slider2.setRange(0, 255)
        self.canny_slider2.setValue(200)
        self.canny_slider2.valueChanged.connect(lambda v: self.update_canny(2, v))
        
        canny_row2.addWidget(self.canny_label2)
        canny_row2.addWidget(self.canny_slider2)
        
        canny_layout.addLayout(canny_row1)
        canny_layout.addLayout(canny_row2)
        canny_group.setLayout(canny_layout)
        edge_layout.addWidget(canny_group)
        
        edge_layout.addStretch()
        self.param_tab.addTab(self.edge_params, "边缘参数")
        
        # 形态学操作参数页（保持不变）
        self.morphology_params = QWidget()
        morphology_layout = QVBoxLayout(self.morphology_params)
        morphology_layout.setContentsMargins(10, 10, 10, 10)
        morphology_layout.setSpacing(12)
        
        # 腐蚀/膨胀参数
        morph_group = QGroupBox("形态学操作参数")
        morph_layout = QVBoxLayout()
        morph_layout.setContentsMargins(8, 8, 8, 8)
        morph_layout.setSpacing(8)
        
        # 结构元素大小
        kernel_row = QHBoxLayout()
        kernel_row.setSpacing(10)
        kernel_row.addWidget(QLabel("结构元素大小:"))
        self.morph_kernel = QSpinBox()
        self.morph_kernel.setRange(1, 9)
        self.morph_kernel.setSingleStep(2)
        self.morph_kernel.setValue(3)
        self.morph_kernel.valueChanged.connect(self.on_param_changed)
        kernel_row.addWidget(self.morph_kernel)
        morph_layout.addLayout(kernel_row)
        
        # 迭代次数
        iter_row = QHBoxLayout()
        iter_row.setSpacing(10)
        iter_row.addWidget(QLabel("迭代次数:"))
        self.morph_iter = QSpinBox()
        self.morph_iter.setRange(1, 5)
        self.morph_iter.setValue(1)
        self.morph_iter.valueChanged.connect(self.on_param_changed)
        iter_row.addWidget(self.morph_iter)
        morph_layout.addLayout(iter_row)
        
        morph_group.setLayout(morph_layout)
        morphology_layout.addWidget(morph_group)
        
        morphology_layout.addStretch()
        self.param_tab.addTab(self.morphology_params, "形态学参数")
        
        # 调整参数面板高度
        self.param_tab.setMinimumHeight(320)
        control_layout.addWidget(self.param_tab)
        
        # 按钮区域（垂直排列，加大按钮）
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        
        # 应用按钮
        self.apply_btn = QPushButton("应用处理")
        self.apply_btn.clicked.connect(self.process_image)
        self.apply_btn.setStyleSheet("""
            padding: 10px; 
            background-color: #4CAF50; 
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 11px;
        """)
        self.apply_btn.setMinimumHeight(40)
        
        # 保存按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setStyleSheet("""
            padding: 10px; 
            background-color: #2196F3; 
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 11px;
        """)
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setEnabled(False)
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.save_btn)
        
        # 底部添加空白区域，避免控件贴底
        button_layout.addSpacing(20)
        control_layout.addLayout(button_layout)
        
        # 将控制面板添加到滚动区域
        control_scroll.setWidget(control_panel)
        main_layout.addWidget(control_scroll)
        
        # 右侧图像显示区域
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(5, 5, 5, 5)
        
        # 图像显示网格
        image_grid = QGridLayout()
        image_grid.setSpacing(15)
        
        # 原始图像显示
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        
        self.original_display = QLabel()
        self.original_display.setAlignment(Qt.AlignCenter)
        self.original_display.setStyleSheet("""
            border: 1px solid #ccc; 
            background-color: #f9f9f9;
            min-height: 320px;
        """)
        
        # 处理后图像显示
        self.processed_label = QLabel("处理后图像")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        
        self.processed_display = QLabel()
        self.processed_display.setAlignment(Qt.AlignCenter)
        self.processed_display.setStyleSheet("""
            border: 1px solid #ccc; 
            background-color: #f9f9f9;
            min-height: 320px;
        """)
        
        # 添加到网格
        image_grid.addWidget(self.original_label, 0, 0)
        image_grid.addWidget(self.original_display, 1, 0)
        image_grid.addWidget(self.processed_label, 0, 1)
        image_grid.addWidget(self.processed_display, 1, 1)
        
        # 设置网格比例
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)
        image_grid.setRowStretch(1, 1)
        
        display_layout.addLayout(image_grid)
        main_layout.addWidget(display_panel, 1)
        
        # 状态栏
        self.statusBar().setStyleSheet("font-size: 10px; padding: 4px;")
        self.statusBar().showMessage("就绪")
        
        self.show()
    
    def load_image(self):
        """加载图像文件（支持中文路径）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if not file_path:
            return
            
        try:
            # 支持中文路径读取
            self.original_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.original_image is None:
                raise Exception("无法解析图像文件")
                
            self.image_path = file_path
            self.processed_image = self.original_image.copy()
            
            # 显示原始图像
            self.display_image(self.original_image, self.original_display)
            
            # 初始显示原始图像
            self.operation_combo.setCurrentText("原始图像")
            self.display_image(self.processed_image, self.processed_display)
            
            # 更新信息
            h, w = self.original_image.shape[:2]
            self.image_info.setText(f"{os.path.basename(file_path)}\n尺寸: {w}×{h}")
            self.statusBar().showMessage(f"已加载图像: {os.path.basename(file_path)}")
            
            # 启用保存按钮
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
            self.statusBar().showMessage("加载图像失败")
    
    def display_image(self, cv_image, qt_label):
        """在QLabel上显示OpenCV图像"""
        if cv_image is None:
            return
            
        # 转换颜色空间
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 获取图像尺寸和标签尺寸
        h, w, c = rgb_image.shape
        label_w = qt_label.width()
        label_h = qt_label.height()
        
        # 计算缩放比例
        scale = min(label_w / w, label_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 调整图像大小
        resized_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为QImage
        q_image = QImage(resized_image.data, new_w, new_h, new_w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 显示图像
        qt_label.setPixmap(pixmap)
    
    def on_operation_changed(self, index):
        """处理操作选择变化"""
        operation = self.operations[index]
        self.statusBar().showMessage(f"已选择: {operation}")
        
        # 切换到对应的参数标签页
        if operation in ["原始图像", "灰度图转换", "图像反转", 
                         "01阈值分割", "均值滤波", "中值滤波"]:
            self.param_tab.setCurrentIndex(0)  # 基础参数
        elif operation in ["Sobel边缘检测", "Canny边缘检测"]:
            self.param_tab.setCurrentIndex(1)  # 边缘参数
        elif operation in ["腐蚀操作", "膨胀操作"]:
            self.param_tab.setCurrentIndex(2)  # 形态学参数
            
        # 应用处理
        self.process_image()
    
    def update_threshold(self, value):
        """更新阈值显示"""
        self.threshold_label.setText(f"阈值: {value}")
        if self.operation_combo.currentText() == "01阈值分割":
            self.process_image()
    
    def update_canny(self, num, value):
        """更新Canny阈值显示"""
        if num == 1:
            self.canny_label1.setText(f"阈值1: {value}")
        else:
            self.canny_label2.setText(f"阈值2: {value}")
            
        if self.operation_combo.currentText() == "Canny边缘检测":
            self.process_image()
    
    def on_param_changed(self):
        """参数变化时处理"""
        current_op = self.operation_combo.currentText()
        if (current_op in ["均值滤波", "中值滤波", "Sobel边缘检测"] or 
            current_op in ["腐蚀操作", "膨胀操作"]):
            self.process_image()
    
    def process_image(self):
        """处理图像"""
        if self.original_image is None:
            return
            
        operation = self.operation_combo.currentText()
        self.statusBar().showMessage(f"正在处理: {operation}")
        
        try:
            if operation == "原始图像":
                self.processed_image = self.original_image.copy()
                
            elif operation == "灰度图转换":
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
            elif operation == "均值滤波":
                ksize = self.blur_kernel.value()
                self.processed_image = cv2.blur(self.original_image, (ksize, ksize))
                
            elif operation == "中值滤波":
                ksize = self.median_kernel.value()
                self.processed_image = cv2.medianBlur(self.original_image, ksize)
                
            elif operation == "Sobel边缘检测":
                ksize = self.sobel_kernel.value()
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                sobel_combined = cv2.addWeighted(
                    cv2.convertScaleAbs(sobelx), 0.5,
                    cv2.convertScaleAbs(sobely), 0.5, 0
                )
                self.processed_image = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
                
            elif operation == "Canny边缘检测":
                th1 = self.canny_slider1.value()
                th2 = self.canny_slider2.value()
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                canny = cv2.Canny(gray, th1, th2)
                self.processed_image = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
                
            elif operation == "01阈值分割":
                th = self.threshold_slider.value()
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                if self.thresh_binary.isChecked():
                    _, threshold = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
                else:
                    _, threshold = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)
                self.processed_image = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
                
            elif operation == "图像反转":
                self.processed_image = cv2.bitwise_not(self.original_image)
                
            elif operation == "腐蚀操作":
                ksize = self.morph_kernel.value()
                iterations = self.morph_iter.value()
                kernel = np.ones((ksize, ksize), np.uint8)
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                eroded = cv2.erode(gray, kernel, iterations=iterations)
                self.processed_image = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
                
            elif operation == "膨胀操作":
                ksize = self.morph_kernel.value()
                iterations = self.morph_iter.value()
                kernel = np.ones((ksize, ksize), np.uint8)
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                dilated = cv2.dilate(gray, kernel, iterations=iterations)
                self.processed_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            
            # 显示处理后的图像
            self.display_image(self.processed_image, self.processed_display)
            self.statusBar().showMessage(f"处理完成: {operation}")
            
        except Exception as e:
            QMessageBox.critical(self, "处理错误", f"处理图像时出错: {str(e)}")
            self.statusBar().showMessage("处理图像失败")
    
    def save_image(self):
        """保存处理后的图像（支持中文路径）"""
        if self.processed_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
            
        # 生成默认文件名
        if self.image_path:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = f"{base_name}_processed.png"
        else:
            default_name = "processed_image.png"
            
        # 获取保存路径
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", default_name, "PNG文件 (*.png);;JPG文件 (*.jpg);;所有文件 (*)"
        )
        
        if not save_path:
            return
            
        try:
            # 支持中文路径保存
            ext = os.path.splitext(save_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            else:  # 默认PNG
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
                
            # 编码并保存
            result, buffer = cv2.imencode(ext, self.processed_image, encode_param)
            if result:
                with open(save_path, 'wb') as f:
                    f.write(buffer)
                
                self.statusBar().showMessage(f"图像已保存至: {save_path}")
                QMessageBox.information(self, "成功", f"图像已成功保存至:\n{save_path}")
            else:
                raise Exception("无法编码图像数据")
                
        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存图像时出错: {str(e)}")
            self.statusBar().showMessage("保存图像失败")
    
    def resizeEvent(self, event):
        """窗口大小变化时重绘图像"""
        super().resizeEvent(event)
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_display)
            self.display_image(self.processed_image, self.processed_display)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用现代风格
    window = ImageProcessor()
    sys.exit(app.exec_())