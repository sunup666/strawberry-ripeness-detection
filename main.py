import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
class StrawberryDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("草莓成熟度智能检测系统")
        self.root.geometry("1200x800")
        
        # 加载训练好的模型
        self.model_path = "runs/detect/train/weights/best.pt"  # 修改为你的模型路径
        self.model = YOLO(self.model_path)
        
        # 初始化摄像头和状态变量
        self.cap = None
        self.is_camera_on = False
        self.detection_enabled = False
        self.confidence_threshold = 0.5
        self.detection_history = []
        self.current_frame = None
        
        # 创建界面
        self.create_widgets()
        
        # 更新摄像头图像
        self.update_camera()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧摄像头和检测区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 摄像头显示区域
        self.camera_canvas = tk.Canvas(left_frame, bg='black')
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 控制面板
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # 摄像头控制按钮
        self.camera_btn = ttk.Button(control_frame, text="开启摄像头", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        self.detect_btn = ttk.Button(control_frame, text="开始检测", command=self.toggle_detection, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        # 置信度阈值调整
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(threshold_frame, text="置信度阈值:").pack(side=tk.LEFT)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0.1, to=0.9, value=0.5, 
                                        command=lambda v: self.set_confidence_threshold(float(v)))
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text="0.5")
        self.threshold_label.pack(side=tk.LEFT)
        
        # 截图按钮
        self.screenshot_btn = ttk.Button(control_frame, text="截图保存", command=self.save_screenshot, state=tk.DISABLED)
        self.screenshot_btn.pack(side=tk.LEFT, padx=5)
        
        # 右侧信息面板
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # 检测统计区域
        stats_group = ttk.LabelFrame(right_frame, text="检测统计", padding=10)
        stats_group.pack(fill=tk.X, pady=5)
        
        self.mature_label = ttk.Label(stats_group, text="成熟草莓: 0", font=('Arial', 10))
        self.mature_label.pack(anchor=tk.W, pady=2)
        
        self.immature_label = ttk.Label(stats_group, text="未成熟草莓: 0", font=('Arial', 10))
        self.immature_label.pack(anchor=tk.W, pady=2)
        
        self.total_label = ttk.Label(stats_group, text="总计: 0", font=('Arial', 10, 'bold'))
        self.total_label.pack(anchor=tk.W, pady=2)
        
        # 成熟度比例显示
        self.ratio_label = ttk.Label(stats_group, text="成熟比例: 0%", font=('Arial', 10))
        self.ratio_label.pack(anchor=tk.W, pady=2)
        
        # 智能建议区域
        advice_group = ttk.LabelFrame(right_frame, text="智能建议", padding=10)
        advice_group.pack(fill=tk.BOTH, pady=5, expand=True)
        
        self.advice_text = tk.Text(advice_group, height=10, width=30, wrap=tk.WORD, 
                                 font=('Arial', 10), state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(advice_group, command=self.advice_text.yview)
        self.advice_text.config(yscrollcommand=scrollbar.set)
        
        self.advice_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 历史数据分析区域
        history_group = ttk.LabelFrame(right_frame, text="成熟度趋势", padding=10)
        history_group.pack(fill=tk.BOTH, pady=5, expand=True)
        
        self.figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=history_group)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 历史记录按钮
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.history_btn = ttk.Button(btn_frame, text="查看历史记录", command=self.show_history)
        self.history_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.export_btn = ttk.Button(btn_frame, text="导出数据", command=self.export_data)
        self.export_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        # 退出按钮
        exit_btn = ttk.Button(right_frame, text="退出系统", command=self.on_close)
        exit_btn.pack(pady=5)
    
    def set_confidence_threshold(self, value):
        self.confidence_threshold = round(value, 1)
        self.threshold_label.config(text=f"{self.confidence_threshold:.1f}")
    
    def toggle_camera(self):
        if self.is_camera_on:
            # 关闭摄像头
            self.stop_camera()
            self.camera_btn.config(text="开启摄像头")
            self.detect_btn.config(state=tk.DISABLED, text="开始检测")
            self.screenshot_btn.config(state=tk.DISABLED)
            self.detection_enabled = False
        else:
            # 开启摄像头
            self.start_camera()
            if self.is_camera_on:
                self.camera_btn.config(text="关闭摄像头")
                self.detect_btn.config(state=tk.NORMAL)
                self.screenshot_btn.config(state=tk.NORMAL)
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            self.is_camera_on = False
            return False
        self.is_camera_on = True
        return True
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.is_camera_on = False
        # 清空画布
        self.camera_canvas.delete("all")
    
    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.detect_btn.config(text="停止检测")
            # 清空历史记录
            self.detection_history = []
        else:
            self.detect_btn.config(text="开始检测")
            # 更新图表
            self.update_chart()
    
    def update_camera(self):
        if self.is_camera_on and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # 如果检测开启，进行目标检测
                if self.detection_enabled:
                    frame, results = self.detect_objects(frame)
                    self.update_stats(results)
                    self.update_advice(results)
                
                # 显示图像
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # 更新画布
                self.camera_canvas.delete("all")
                self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.camera_canvas.image = img_tk
        
        # 每15毫秒更新一次
        self.root.after(15, self.update_camera)
    
    def detect_objects(self, frame):
    # 使用YOLOv8进行检测
        results = self.model(frame, conf=self.confidence_threshold)
    
    # 记录检测结果
        timestamp = datetime.now()
        mature_count = 0
        immature_count = 0
    
    # 绘制检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
            # 获取坐标和类别
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
            
            # 统计数量
                if cls == 0:  # 成熟
                   mature_count += 1
                   color = (0, 255, 0)  # 绿色
                   label = f"ripe0 ({conf:.2f})"
                else:  # 未成熟
                   immature_count += 1
                   color = (0, 0, 255)  # 红色
                   label = f"unripe1 ({conf:.2f})"
            
            # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 计算标签文本大小
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # 绘制标签背景
                cv2.rectangle(frame, 
                             (x1, y1 - label_height - 10), 
                             (x1 + label_width, y1), 
                              color, -1)
            
            # 绘制标签文本
                cv2.putText(frame, label, 
                           (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)
    
    # 记录检测结果
        self.detection_history.append({
           "timestamp": timestamp,
           "mature": mature_count,
           "immature": immature_count,
           "total": mature_count + immature_count
           })
    
        return frame, (mature_count, immature_count)
    
    def update_stats(self, results):
        mature, immature = results
        total = mature + immature
        ratio = mature / total * 100 if total > 0 else 0
        
        self.mature_label.config(text=f"成熟草莓: {mature}")
        self.immature_label.config(text=f"未成熟草莓: {immature}")
        self.total_label.config(text=f"总计: {total}")
        self.ratio_label.config(text=f"成熟比例: {ratio:.1f}%")
    
    def update_advice(self, results):
        mature, immature = results
        total = mature + immature
        
        advice = []
        if total == 0:
            advice.append("⚠️ 未检测到草莓")
            advice.append("请调整摄像头位置或检测参数")
        else:
            mature_ratio = mature / total
            
            # 成熟度建议
            if mature_ratio > 0.7:
                advice.append("✅ 检测到大量成熟草莓")
                advice.append("建议立即安排采摘和销售")
            elif mature_ratio > 0.4:
                advice.append("🟡 草莓成熟度适中")
                advice.append("建议近期安排采摘")
            else:
                advice.append("🔴 大部分草莓未成熟")
                advice.append("建议继续生长观察")
            
            # 特殊情况的额外建议
            if immature > 10 and mature < 3:
                advice.append("\n⚠️ 注意：当前未成熟草莓较多")
                advice.append("可能需要调整种植条件")
            elif mature > 15:
                advice.append("\n⚠️ 注意：成熟草莓数量较多")
                advice.append("请尽快采摘以避免过熟")
            
            # 数量建议
            if total > 20:
                advice.append("\nℹ️ 当前草莓数量较多")
                advice.append("建议分批采摘")
        
        self.advice_text.config(state=tk.NORMAL)
        self.advice_text.delete(1.0, tk.END)
        self.advice_text.insert(tk.END, "\n".join(advice))
        self.advice_text.config(state=tk.DISABLED)
    
    def update_chart(self):
        if not self.detection_history:
            return
        
        # 准备数据
        timestamps = [d["timestamp"] for d in self.detection_history]
        mature = [d["mature"] for d in self.detection_history]
        immature = [d["immature"] for d in self.detection_history]
        
        # 清空图表
        self.ax.clear()
        
        # 绘制堆叠面积图
        self.ax.stackplot(timestamps, mature, immature, 
                         labels=['成熟', '未成熟'],
                         colors=['#4CAF50', '#F44336'], alpha=0.7)
        
        # 绘制折线图
        self.ax.plot(timestamps, mature, color='#2E7D32', linewidth=2, label='_成熟')
        self.ax.plot(timestamps, immature, color='#C62828', linewidth=2, label='_未成熟')
        
        self.ax.set_xlabel('检测时间')
        self.ax.set_ylabel('草莓数量')
        self.ax.set_title('草莓成熟度趋势变化')
        self.ax.legend(loc='upper left')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # 自动调整日期显示
        self.figure.autofmt_xdate()
        
        # 更新画布
        self.canvas.draw()
    
    def show_history(self):
        if not self.detection_history:
            messagebox.showinfo("提示", "暂无历史检测数据")
            return
        
        # 创建历史记录窗口
        history_window = tk.Toplevel(self.root)
        history_window.title("检测历史记录")
        history_window.geometry("900x500")
        
        # 创建表格框架
        frame = ttk.Frame(history_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建滚动条
        scroll_y = ttk.Scrollbar(frame)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建表格
        columns = ("时间", "成熟数量", "未成熟数量", "总计", "成熟比例")
        self.history_tree = ttk.Treeview(frame, columns=columns, 
                                        yscrollcommand=scroll_y.set,
                                        xscrollcommand=scroll_x.set,
                                        show="headings")
        
        # 配置列
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=150, anchor=tk.CENTER)
        
        # 添加数据
        for record in self.detection_history:
            total = record["mature"] + record["immature"]
            ratio = record["mature"] / total * 100 if total > 0 else 0
            self.history_tree.insert("", tk.END, values=(
                record["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                record["mature"],
                record["immature"],
                total,
                f"{ratio:.1f}%"
            ))
        
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
        scroll_y.config(command=self.history_tree.yview)
        scroll_x.config(command=self.history_tree.xview)
    
    def export_data(self):
        if not self.detection_history:
            messagebox.showinfo("提示", "暂无数据可导出")
            return
        
        # 转换为DataFrame
        data = []
        for record in self.detection_history:
            total = record["mature"] + record["immature"]
            ratio = record["mature"] / total * 100 if total > 0 else 0
            data.append({
                "时间": record["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "成熟数量": record["mature"],
                "未成熟数量": record["immature"],
                "总计": total,
                "成熟比例(%)": ratio
            })
        
        df = pd.DataFrame(data)
        
        # 选择保存路径
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx")],
            initialfile=f"草莓检测数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if filename:
            if filename.endswith('.csv'):
                df.to_csv(filename, index=False, encoding='utf_8_sig')
            else:
                df.to_excel(filename, index=False)
            
            messagebox.showinfo("成功", f"数据已导出到:\n{filename}")
    
    def save_screenshot(self):
        if self.current_frame is None:
            messagebox.showwarning("警告", "没有可保存的图像")
            return
        
        # 选择保存路径
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG图像", "*.jpg"), ("PNG图像", "*.png")],
            initialfile=f"草莓检测_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if filename:
            # 转换颜色空间并保存
            img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, img)
            messagebox.showinfo("成功", f"截图已保存到:\n{filename}")
    
    def on_close(self):
        if messagebox.askokcancel("退出", "确定要退出系统吗？"):
            if self.cap:
                self.cap.release()
            self.root.destroy()

if __name__ == "__main__":
    # 检查模型文件是否存在
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        print("请确保已经训练好模型并放在正确路径")
    else:
        root = tk.Tk()
        app = StrawberryDetectionApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()