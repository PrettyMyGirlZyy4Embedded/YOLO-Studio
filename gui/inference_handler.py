import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk, ImageDraw
import random
import threading, queue, shutil, tempfile
import cv2

try:
    from security import is_pro_version
except ImportError:
    def is_pro_version(): return False

class InferenceHandler:
    def __init__(self, app):
        self.app = app
        self.result_image = None
        self.progress_queue = queue.Queue()
        self.infer_thread = None
        self.video_path = None
        self.video_playing = False
        self.cap = None

    def draw_boxes_on_image(self, image_path, boxes, labels=None, scores=None, color=(255,0,0)):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = ""
            if labels is not None:
                label += str(labels[i])
            if scores is not None:
                label += f" {scores[i]:.2f}"
            if label:
                draw.text((x1, max(0, y1-15)), label, fill=color)
        return img

    def select_model(self):
        if not is_pro_version():
            messagebox.showwarning("专业版功能", "推理功能为专业版专属，请激活专业版。", parent=self.app.master)
            return
        file_path = filedialog.askopenfilename(
            title="选择推理模型文件",
            filetypes=[("PyTorch/ONNX/TFLite/OpenVINO模型", "*.pt *.onnx *.tflite *.xml"), ("所有文件", "*.*")],
            parent=self.app.master
        )
        if file_path:
            self.app.infer_model_path_var.set(file_path)

    def select_input_file(self):
        if not is_pro_version():
            messagebox.showwarning("专业版功能", "推理功能为专业版专属，请激活专业版。", parent=self.app.master)
            return
        file_path = filedialog.askopenfilename(
            title="选择图片或视频文件",
            filetypes=[("图片/视频", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov"), ("所有文件", "*.*")],
            parent=self.app.master
        )
        if file_path:
            self.app.infer_input_file_var.set(file_path)

    def _safe_video_path(self, src_path):
        # 复制到临时英文目录，避免中文/特殊字符/空格
        if not src_path.lower().endswith(('.mp4', '.avi', '.mov')):
            return src_path
        temp_dir = os.path.join(tempfile.gettempdir(), "yolo_infer")
        os.makedirs(temp_dir, exist_ok=True)
        dst_path = os.path.join(temp_dir, "infer.mp4")
        try:
            shutil.copy2(src_path, dst_path)
            return dst_path
        except Exception:
            return src_path  # 复制失败则用原路径

    def start_inference(self):
        if not is_pro_version():
            messagebox.showwarning("专业版功能", "推理功能为专业版专属，请激活专业版。", parent=self.app.master)
            return
        model_path = self.app.infer_model_path_var.get()
        input_file = self.app.infer_input_file_var.get()
        conf = self.app.infer_conf_var.get()
        iou = self.app.infer_iou_var.get()
        yolo_code_path = self.app.global_yolo_code_path.get() if hasattr(self.app, 'global_yolo_code_path') else None
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("错误", "请先选择有效的模型文件。", parent=self.app.master)
            return
        if not input_file or not os.path.exists(input_file):
            messagebox.showerror("错误", "请先选择有效的图片或视频文件。", parent=self.app.master)
            return
        self.app.infer_progress_label.config(text="推理中...", foreground="#a6adc8")
        self.app.infer_progress_bar['value'] = 10
        self.app.infer_output_text.config(state="normal")
        self.app.infer_output_text.delete(1.0, tk.END)
        self.app.infer_output_text.insert(tk.END, f"模型: {model_path}\n输入: {input_file}\n置信度: {conf}  IoU: {iou}\n开始推理...\n")
        self.app.infer_output_text.config(state="disabled")
        self.app.master.update_idletasks()
        self.app.start_infer_btn.config(state="disabled")
        # 启动子线程
        self.infer_thread = threading.Thread(target=self._run_inference_thread, args=(model_path, input_file, conf, iou, yolo_code_path), daemon=True)
        self.infer_thread.start()
        self._update_progress_bar()

    def _update_progress_bar(self):
        try:
            while not self.progress_queue.empty():
                percent, msg = self.progress_queue.get_nowait()
                self.app.infer_progress_bar['value'] = percent
                self.app.infer_progress_label.config(text=msg)
        except queue.Empty:
            pass
        if self.infer_thread and self.infer_thread.is_alive():
            self.app.master.after(100, self._update_progress_bar)
        else:
            self.app.infer_progress_bar['value'] = 100

    def _run_inference_thread(self, model_path, input_file, conf, iou, yolo_code_path):
        try:
            import sys
            import importlib
            import torch
            import numpy as np
            from PIL import Image
            # 动态导入YOLO代码
            if yolo_code_path and os.path.isdir(yolo_code_path):
                if yolo_code_path not in sys.path:
                    sys.path.insert(0, yolo_code_path)
            # 优先尝试ultralytics YOLO
            try:
                ultralytics = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics, "YOLO", None)
            except Exception:
                YOLO = None
            results = None
            if YOLO is not None:
                model = YOLO(model_path)
                results = model(input_file, conf=float(conf), iou=float(iou))
                boxes = []
                labels = []
                scores = []
                for r in results:
                    if hasattr(r, 'boxes') and hasattr(r.boxes, 'xyxy'):
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else None
                        clss = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else None
                        for i, box in enumerate(xyxy):
                            boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                            score = float(confs[i]) if confs is not None else 1.0
                            scores.append(score)
                            label = str(int(clss[i])) if clss is not None else "目标"
                            labels.append(label)
                img_boxed = self.draw_boxes_on_image(input_file, boxes, labels, scores)
                img_boxed = img_boxed.resize((640, 480))
                self.app.master.after(0, self._on_inference_done, img_boxed, "推理完成，检测框已绘制。")
            else:
                # 兼容YOLOv5 detect.py
                import subprocess
                out_dir = tempfile.mkdtemp()
                py_exec = sys.executable
                detect_py = os.path.join(yolo_code_path, "detect.py")
                # 视频路径特殊处理
                safe_input = self._safe_video_path(input_file)
                # 命令参数全部加引号
                cmd = [f'"{py_exec}"', f'"{detect_py}"', '--weights', f'"{model_path}"', '--source', f'"{safe_input}"', '--conf', str(conf), '--iou', str(iou), '--project', f'"{out_dir}"', '--name', 'infer', '--exist-ok']
                self.progress_queue.put((20, "正在调用YOLO detect.py..."))
                def log_cmd():
                    self.app.infer_output_text.config(state="normal")
                    self.app.infer_output_text.insert(tk.END, f"\n调用命令: {' '.join(cmd)}\n")
                    self.app.infer_output_text.config(state="disabled")
                    self.app.master.update_idletasks()
                self.app.master.after(0, log_cmd)
                subprocess.run(' '.join(cmd), shell=True, check=True)
                self.progress_queue.put((60, "推理完成，正在查找输出..."))
                # 查找输出图片/视频
                infer_dir = os.path.join(out_dir, 'infer')
                found = False
                img_boxed = None
                video_path = None
                for f in os.listdir(infer_dir):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                        img_path = os.path.join(infer_dir, f)
                        img_boxed = Image.open(img_path).resize((640, 480))
                        found = True
                        break
                    if f.lower().endswith(('.mp4', '.avi', '.mov')):
                        # 视频推理结果，取第一帧做展示
                        video_path = os.path.join(infer_dir, f)
                        cap = cv2.VideoCapture(video_path)
                        ret, frame = cap.read()
                        if ret:
                            from PIL import Image
                            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            img_boxed = img.resize((640, 480))
                            found = True
                        cap.release()
                        break
                if not found:
                    raise Exception("未找到推理输出图片或视频")
                if video_path:
                    # 推理结果为视频，主线程展示提示和打开按钮
                    self.app.master.after(0, self._on_inference_video, img_boxed, video_path)
                else:
                    self.app.master.after(0, self._on_inference_done, img_boxed, "推理完成，检测框已绘制。")
        except Exception as e:
            self.app.master.after(0, self._on_inference_error, str(e))

    def _on_inference_done(self, img_boxed, msg):
        # 图片推理结果，显示到Canvas
        self.result_image = ImageTk.PhotoImage(img_boxed)
        canvas = self.app.infer_result_canvas
        canvas.delete("all")
        canvas.create_image(0, 0, image=self.result_image, anchor="nw")
        canvas.image = self.result_image
        self.app.infer_progress_label.config(text="推理完成！", foreground="#a6e3a1")
        self.app.infer_progress_bar['value'] = 100
        self.app.infer_output_text.config(state="normal")
        self.app.infer_output_text.insert(tk.END, msg + "\n")
        self.app.infer_output_text.config(state="disabled")
        self.app.start_infer_btn.config(state="normal")

    def _on_inference_video(self, img_boxed, video_path):
        # 视频推理结果，自动播放视频到Canvas
        self.video_path = video_path
        self.video_playing = True
        self.cap = cv2.VideoCapture(video_path)
        self._play_video_frame()
        self.app.infer_progress_label.config(text="推理完成！(视频)", foreground="#a6e3a1")
        self.app.infer_progress_bar['value'] = 100
        self.app.infer_output_text.config(state="normal")
        self.app.infer_output_text.insert(tk.END, f"推理完成，检测框已绘制。\n推理视频已生成：{video_path}\n")
        self.app.infer_output_text.config(state="disabled")
        self.app.start_infer_btn.config(state="normal")
        # 增加播放/暂停按钮
        def toggle_play():
            self.video_playing = not self.video_playing
            if self.video_playing:
                self._play_video_frame()
        if not hasattr(self.app, '_infer_video_play_btn'):
            from tkinter import ttk
            self.app._infer_video_play_btn = ttk.Button(self.app.infer_result_display_frame, text="暂停/播放", command=toggle_play)
            self.app._infer_video_play_btn.grid(row=1, column=0, sticky="ew", pady=(5,0))
        else:
            self.app._infer_video_play_btn.config(command=toggle_play)
        self.app._infer_video_play_btn.lift()
        # 增加打开文件夹按钮
        def open_folder():
            import subprocess, os
            folder = os.path.dirname(video_path)
            if os.name == 'nt':
                os.startfile(folder)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', folder])
        if not hasattr(self.app, '_infer_open_folder_btn'):
            from tkinter import ttk
            self.app._infer_open_folder_btn = ttk.Button(self.app.infer_result_display_frame, text="打开视频所在文件夹", command=open_folder)
            self.app._infer_open_folder_btn.grid(row=2, column=0, sticky="ew", pady=(5,0))
        else:
            self.app._infer_open_folder_btn.config(command=open_folder)
        self.app._infer_open_folder_btn.lift()

    def _play_video_frame(self):
        if not hasattr(self, 'cap') or self.cap is None:
            return
        if not self.video_playing:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(img)
            canvas = self.app.infer_result_canvas
            canvas.delete("all")
            canvas.create_image(0, 0, image=imgtk, anchor="nw")
            canvas.image = imgtk
            self.app.master.after(33, self._play_video_frame)  # 约30fps
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._play_video_frame()

    def _on_inference_error(self, msg):
        # 在Canvas上显示错误信息
        canvas = self.app.infer_result_canvas
        canvas.delete("all")
        canvas.create_text(320, 240, text="推理失败", fill="#f38ba8", font=("Microsoft YaHei UI", 24, "bold"))
        self.app.infer_progress_label.config(text="推理失败", foreground="#f38ba8")
        self.app.infer_progress_bar['value'] = 0
        self.app.infer_output_text.config(state="normal")
        self.app.infer_output_text.insert(tk.END, f"推理失败: {msg}\n")
        self.app.infer_output_text.config(state="disabled")
        self.app.start_infer_btn.config(state="normal") 