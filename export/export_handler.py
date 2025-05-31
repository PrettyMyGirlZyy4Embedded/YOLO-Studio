# export/export_handler.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog,ttk
import os
import sys
import subprocess
import threading
import platform
import re
import json # For parsing potential dicts from string
from config import config as global_config

# 导入安全模块
try:
    from security import protect_export_feature, check_feature_access
except ImportError:
    def protect_export_feature(func): return func
    def check_feature_access(feature_name): return False

# Try to import conversion-specific libraries within functions to avoid hard dependencies
# if the user doesn't use those specific export features.

class ExportHandler:
    def __init__(self, app_instance):
        self.app = app_instance
        self.export_formats = {
            "PyTorch (.pt) -> ONNX": {
                "input_ext": ".pt",
                "output_ext": ".onnx",
                "options_func": self._get_onnx_options_ui,
                "export_func": self._export_pt_to_onnx,
                "default_opset": 12,
                "default_dynamic_axes_str": "{'input': {0: 'batch'}, 'output': {0: 'batch'}}"
            },
            "ONNX -> TensorFlow Lite (.tflite)": {
                "input_ext": ".onnx",
                "output_ext": ".tflite",
                "options_func": self._get_tflite_options_ui,
                "export_func": self._export_onnx_to_tflite,
                "tf_optimizations": ["DEFAULT", "OPTIMIZE_FOR_SIZE", "OPTIMIZE_FOR_LATENCY", "NO_OPTIMIZATION"],
            },
            "ONNX -> OpenVINO IR (.xml/.bin)": {
                "input_ext": ".onnx",
                "output_ext": ".xml", # Main file, .bin is also created
                "options_func": self._get_openvino_options_ui,
                "export_func": self._export_onnx_to_openvino,
                "data_types": ["FP32", "FP16"],
            },
            # Placeholder for more:
            # "ONNX -> TensorFlow SavedModel": { ... }
            # "PyTorch (.pt) -> TorchScript (.ptl)": { ... }
        }
        # Keep quantization as a general option, applicability depends on the chosen path
        self.quantization_options = ["None", "FP16", "INT8 (Dynamic)", "INT8 (Static - requires calibration data)"]


    def _log(self, message, is_error=False):
        self.app._add_to_output_log_queue(f"[Export] {message}\n", is_error)

    def _update_progress_label(self, text):
        if hasattr(self.app, 'export_progress_label') and self.app.export_progress_label:
            try:
                # 检查窗口部件是否仍然存在
                if self.app.export_progress_label.winfo_exists():
                    self.app.export_progress_label.config(text=text)
            except tk.TclError:
                # 窗口部件可能已被删除，忽略错误
                pass

    def _set_progress_bar(self, value=None, mode='determinate', start_animation=False):
        if hasattr(self.app, 'export_progress_bar') and self.app.export_progress_bar:
            try:
                # 检查窗口部件是否仍然存在
                if self.app.export_progress_bar.winfo_exists():
                    if mode == 'indeterminate':
                        self.app.export_progress_bar.config(mode='indeterminate')
                        if start_animation: 
                            self.app.export_progress_bar.start(10)
                    else:
                        self.app.export_progress_bar.stop()
                        self.app.export_progress_bar.config(mode='determinate', value=value if value is not None else 0)
            except tk.TclError:
                # 窗口部件可能已被删除，忽略错误
                pass

    def select_input_model(self):
        file_path = filedialog.askopenfilename(
            title="选择输入模型文件",
            filetypes=[("PyTorch models", "*.pt"), ("ONNX models", "*.onnx"), ("All files", "*.*")],
            parent=self.app.master
        )
        if file_path:
            self.app.export_input_model_path.set(file_path)
            base, ext = os.path.splitext(os.path.basename(file_path))
            self.app.export_output_filename.set(base)
            # No auto-selection of format here, let user choose the conversion path explicitly
            self.on_export_format_selected() # To update options if a format is already selected

    def select_output_directory(self):
        dir_path = filedialog.askdirectory(title="选择输出文件夹", parent=self.app.master)
        if dir_path:
            self.app.export_output_dir_path.set(dir_path)

    def select_export_dir(self):
        """选择导出目录"""
        dir_path = filedialog.askdirectory(title="选择导出文件夹", parent=self.app.master)
        if dir_path:
            self.app.export_dir_var.set(dir_path)

    def _clear_dynamic_options_frame(self):
        if hasattr(self.app, 'export_dynamic_options_frame') and self.app.export_dynamic_options_frame is not None and self.app.export_dynamic_options_frame.winfo_exists():
            for widget in self.app.export_dynamic_options_frame.winfo_children():
                widget.destroy()

    def on_export_format_selected(self, event=None):
        self._clear_dynamic_options_frame()
        selected_format_name = self.app.export_selected_format.get()
        format_details = self.export_formats.get(selected_format_name)
        parent_frame = getattr(self.app, 'export_dynamic_options_frame', None)
        if not format_details or parent_frame is None:
            return
        
        options_func = format_details.get("options_func")
        if options_func:
            options_func(parent_frame, format_details)


    def _get_onnx_options_ui(self, parent_frame, format_details):
        if parent_frame is None:
            return
        parent_frame.columnconfigure(1, weight=1)
        label_bg_color = global_config.UI_BACKGROUND_COLOR

        opset_label = tk.Label(parent_frame, text="Opset Version:", bg=label_bg_color)
        opset_label.grid(row=0, column=0, sticky="w", padx=(0,5), pady=2)
        self.app.export_onnx_opset_var.set(str(format_details.get("default_opset", 12)))
        opset_entry = ttk.Entry(parent_frame, textvariable=self.app.export_onnx_opset_var, width=10)
        opset_entry.grid(row=0, column=1, sticky="w", pady=2)

        dyn_axes_label = tk.Label(parent_frame, text="Dynamic Axes (JSON/dict):", bg=label_bg_color)
        dyn_axes_label.grid(row=1, column=0, sticky="w", padx=(0,5), pady=2)
        self.app.export_onnx_dynamic_axes_var.set(format_details.get("default_dynamic_axes_str", ""))
        dyn_axes_entry = ttk.Entry(parent_frame, textvariable=self.app.export_onnx_dynamic_axes_var)
        dyn_axes_entry.grid(row=1, column=1, sticky="ew", pady=2)
        
        # 添加输入尺寸设置
        input_size_label = tk.Label(parent_frame, text="输入尺寸 H,W (如 640,640):", bg=label_bg_color)
        input_size_label.grid(row=2, column=0, sticky="w", padx=(0,5), pady=2)
        if not hasattr(self.app, 'export_onnx_input_size_var'):
            self.app.export_onnx_input_size_var = tk.StringVar(value="640,640")
        input_size_entry = ttk.Entry(parent_frame, textvariable=self.app.export_onnx_input_size_var, width=15)
        input_size_entry.grid(row=2, column=1, sticky="w", pady=2)

    def _get_tflite_options_ui(self, parent_frame, format_details):
        if parent_frame is None:
            return
        parent_frame.columnconfigure(1, weight=1)
        label_bg_color = global_config.UI_BACKGROUND_COLOR

        optimizations = format_details.get("tf_optimizations", ["DEFAULT"])
        opt_label = tk.Label(parent_frame, text="TF Lite Optimizations:", bg=label_bg_color)
        opt_label.grid(row=0, column=0, sticky="w", padx=(0,5), pady=2)
        self.app.export_tflite_optimization_var.set(optimizations[0])
        opt_combo = ttk.Combobox(parent_frame, textvariable=self.app.export_tflite_optimization_var,
                                 values=optimizations, state="readonly", width=25)
        opt_combo.grid(row=0, column=1, sticky="ew", pady=2)

    def _get_openvino_options_ui(self, parent_frame, format_details):
        if parent_frame is None:
            return
        parent_frame.columnconfigure(1, weight=1)
        label_bg_color = global_config.UI_BACKGROUND_COLOR

        data_types = format_details.get("data_types", ["FP32"])
        dt_label = tk.Label(parent_frame, text="Data Type (Precision):", bg=label_bg_color)
        dt_label.grid(row=0, column=0, sticky="w", padx=(0,5), pady=2)
        self.app.export_openvino_precision_var.set(data_types[0])
        dt_combo = ttk.Combobox(parent_frame, textvariable=self.app.export_openvino_precision_var,
                                values=data_types, state="readonly", width=15)
        dt_combo.grid(row=0, column=1, sticky="w", pady=2) # sticky w

        # Input shape (optional, for overriding)
        shape_label = tk.Label(parent_frame, text="Input Shape (e.g., [1,3,640,640]):", bg=label_bg_color)
        shape_label.grid(row=1, column=0, sticky="w", padx=(0,5), pady=2)
        self.app.export_openvino_input_shape_var.set("") # Default empty
        shape_entry = ttk.Entry(parent_frame, textvariable=self.app.export_openvino_input_shape_var)
        shape_entry.grid(row=1, column=1, sticky="ew", pady=2)


    @protect_export_feature
    def start_export(self):
        # 检查是否有权限使用导出功能
        try:
            if not check_feature_access("export_allowed"):
                # 如果没有权限，显示提示并返回
                messagebox.showwarning("许可证限制", 
                                      "模型导出功能需要专业版许可证。\n请升级到专业版解锁此功能。", 
                                      parent=self.app.master)
                return
        except ImportError:
            pass  # 如果安全模块不存在，继续执行
            
        input_path = self.app.export_input_model_path.get()
        output_dir = self.app.export_output_dir_path.get()
        output_filename_base = self.app.export_output_filename.get()
        selected_format_name = self.app.export_selected_format.get()
        quantization_type = self.app.export_quantization_type.get()

        # Validations
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("错误", "请输入有效的输入模型路径。", parent=self.app.master); return
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("错误", "请选择有效的输出文件夹。", parent=self.app.master); return
        if not output_filename_base:
            messagebox.showerror("错误", "请输入输出文件名。", parent=self.app.master); return
        if not selected_format_name or selected_format_name not in self.export_formats:
            messagebox.showerror("错误", "请选择有效的导出格式。", parent=self.app.master); return

        format_details = self.export_formats[selected_format_name]
        
        # Validate input file extension against selected format path
        _, input_ext_actual = os.path.splitext(input_path)
        if input_ext_actual.lower() != format_details["input_ext"].lower():
            messagebox.showerror("输入错误", f"所选导出路径 '{selected_format_name}' 需要 '{format_details['input_ext']}' 输入文件，但提供了 '{input_ext_actual}' 文件。", parent=self.app.master)
            return

        output_filename = output_filename_base + format_details["output_ext"]
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            if not messagebox.askyesno("确认", f"文件 {output_filename} 已存在。要覆盖吗？", parent=self.app.master):
                return

        self.app.start_export_btn.config(state="disabled")
        self._update_progress_label(f"准备导出: {selected_format_name}...")
        self._set_progress_bar(mode='indeterminate', start_animation=True)

        export_options = {
            "input_path": input_path,
            "output_path": output_path, # Base output path, some tools create multiple files
            "output_dir": output_dir,
            "output_filename_base": output_filename_base,
            "selected_format": selected_format_name,
            "quantization": quantization_type,
        }

        # Gather format-specific options
        if selected_format_name == "PyTorch (.pt) -> ONNX":
            try: 
                export_options["opset_version"] = int(self.app.export_onnx_opset_var.get())
            except ValueError: 
                messagebox.showerror("错误", "ONNX Opset Version 必须是整数。", parent=self.app.master)
                self._export_finished(False, "Opset无效")
                return
                
            # 修改动态轴参数的处理逻辑，使其成为可选参数
            dynamic_axes_str = self.app.export_onnx_dynamic_axes_var.get().strip()
            if dynamic_axes_str:
                try:
                    # 尝试解析为JSON
                    export_options["dynamic_axes"] = json.loads(dynamic_axes_str.replace("'", "\""))
                    if not isinstance(export_options["dynamic_axes"], dict):
                        raise ValueError("动态轴必须是字典格式")
                except Exception as e:
                    self._log(f"动态轴参数解析错误: {e}", is_error=True)
                    self._log("将继续导出但不使用动态轴，模型将使用固定输入大小")
                    export_options["dynamic_axes"] = None
            else:
                # 没有提供动态轴参数，这是完全有效的
                export_options["dynamic_axes"] = None
                self._log("未提供动态轴参数，模型将使用固定输入大小")
        elif selected_format_name == "ONNX -> TensorFlow Lite (.tflite)":
            export_options["tflite_optimization"] = self.app.export_tflite_optimization_var.get()
        elif selected_format_name == "ONNX -> OpenVINO IR (.xml/.bin)":
            export_options["openvino_precision"] = self.app.export_openvino_precision_var.get()
            input_shape_str = self.app.export_openvino_input_shape_var.get().strip()
            if input_shape_str:
                try: # Expects string like "[1,3,224,224]" or "1,3,224,224"
                    export_options["openvino_input_shape"] = input_shape_str
                except Exception as e: messagebox.showerror("错误", f"OpenVINO 输入形状格式无效: {e}"); self._export_finished(False, "OpenVINO输入形状无效"); return
            else: export_options["openvino_input_shape"] = None


        threading.Thread(target=self._export_model_thread, args=(export_options,), daemon=True).start()

    def _export_model_thread(self, options):
        try:
            self._log(f"开始转换 {os.path.basename(options['input_path'])} 为 {options['selected_format']}")
            
            export_func = self.export_formats[options["selected_format"]].get("export_func")
            if export_func:
                export_func(options) # Call the specific export function
            else:
                self._log(f"未找到针对 {options['selected_format']} 的导出函数。", is_error=True)
                self._export_finished(success=False, message=f"不支持的格式: {options['selected_format']}")

        except Exception as e:
            self._log(f"导出过程中发生严重错误: {e}", is_error=True)
            import traceback
            self._log(traceback.format_exc(), is_error=True)
            self._export_finished(success=False, message=f"导出失败: {e}")

    def _export_pt_to_onnx(self, options):
        try:
            import torch
            # 添加 YOLO 代码路径到 sys.path
            yolo_code_path = self.app.global_yolo_code_path.get() if hasattr(self.app, 'global_yolo_code_path') else None
            if yolo_code_path and os.path.exists(yolo_code_path):
                sys.path.insert(0, yolo_code_path)
                self._log(f"添加 YOLO 代码路径到 sys.path: {yolo_code_path}")

            # 使用更安全的方式更新UI
            try:
                if self.app.master.winfo_exists():
                    self.app.master.after(0, self._update_progress_label, f"加载PyTorch模型: {os.path.basename(options['input_path'])}")
            except tk.TclError:
                pass
            
            model_data = torch.load(options['input_path'], map_location='cpu')
            if isinstance(model_data, dict) and 'model' in model_data:
                 model = model_data['model'].float().eval() # Ultralytics style
            elif isinstance(model_data, torch.nn.Module): # Directly saved model
                 model = model_data.float().eval()
            else:
                self._log("无法识别的.pt文件结构。需要包含'model'键或直接是torch.nn.Module。", is_error=True)
                self._export_finished(False, ".pt文件结构无法识别")
                return

            # 移除 YOLO 代码路径
            if yolo_code_path in sys.path:
                sys.path.remove(yolo_code_path)

            # 使用更安全的方式更新UI
            try:
                if self.app.master.winfo_exists():
                    self.app.master.after(0, self._update_progress_label, "PyTorch模型加载成功。导出到ONNX...")
            except tk.TclError:
                pass
            
            # 使用预设的输入尺寸，而不是弹出对话框
            img_size_str = None
            
            # 首先尝试从UI获取
            if hasattr(self.app, 'export_onnx_input_size_var'):
                img_size_str = self.app.export_onnx_input_size_var.get()
                self._log(f"使用UI中设置的输入尺寸: {img_size_str}")
            
            # 如果没有设置或无效，使用默认值
            if not img_size_str:
                img_size_str = "640,640"  # 默认尺寸
                self._log(f"使用默认输入尺寸: {img_size_str}")
                
            try:
                h, w = map(int, img_size_str.split(','))
                dummy_input = torch.randn(1, 3, h, w, device='cpu')
                self._log(f"使用随机输入尺寸: (1, 3, {h}, {w})")
            except ValueError: 
                self._log("输入尺寸格式错误，使用默认值640,640")
                h, w = 640, 640
                dummy_input = torch.randn(1, 3, h, w, device='cpu')
                self._log(f"使用随机输入尺寸: (1, 3, {h}, {w})")

            # 安全处理dynamic_axes参数
            dynamic_axes = None
            try:
                if "dynamic_axes" in options and options["dynamic_axes"] is not None:
                    dynamic_axes = options["dynamic_axes"]
                    self._log(f"使用动态轴: {dynamic_axes}")
                else:
                    # 如果未提供或为None，使用默认值或不设置
                    self._log("未指定动态轴，模型将使用固定输入大小")
            except Exception as e:
                self._log(f"解析动态轴时出错: {e}，将使用固定大小导出", is_error=True)
                dynamic_axes = None

            # 导出ONNX
            export_args = {
                'verbose': False,
                'input_names': ['images' if hasattr(model, 'names') else 'input'],
                'output_names': ['output0' if hasattr(model, 'names') else 'output'],
                'opset_version': options.get("opset_version", 12)
            }
            
            # 仅当dynamic_axes有效时添加到参数
            if dynamic_axes is not None:
                export_args['dynamic_axes'] = dynamic_axes

            torch.onnx.export(model, dummy_input, options['output_path'], **export_args)
            self._log(f"成功导出到ONNX: {options['output_path']}")
            
            # Optional ONNX verification
            try:
                import onnx
                onnx_model = onnx.load(options['output_path'])
                onnx.checker.check_model(onnx_model)
                self._log("ONNX模型结构检查通过。")
            except Exception as e: 
                self._log(f"ONNX模型验证失败: {e}", is_error=True)
                
            self._export_finished(True, f"成功导出到 {os.path.basename(options['output_path'])}")
        except ImportError as e:
            self._log(f"导出ONNX失败: 缺少库 {e.name if hasattr(e, 'name') else str(e)}。", True)
            try:
                if self.app.master.winfo_exists():
                    messagebox.showerror("缺少库", f"需要库 '{e.name if hasattr(e, 'name') else str(e)}'。请手动安装。", parent=self.app.master)
            except tk.TclError:
                pass
            self._export_finished(False, f"缺少库: {e.name if hasattr(e, 'name') else str(e)}")
        except Exception as e:
            self._log(f"导出到ONNX时发生错误: {e}", True)
            import traceback
            self._log(traceback.format_exc(), True)
            self._export_finished(False, f"ONNX导出失败: {e}")


    def _run_subprocess_command(self, command_list, step_name):
        """Helper to run a subprocess command and log its output."""
        self._log(f"执行 {step_name}: {' '.join(command_list)}")
        
        # 安全地调用UI更新
        try:
            if self.app.master.winfo_exists():
                self.app.master.after(0, self._update_progress_label, f"正在执行: {step_name}...")
                self.app.master.after(0, lambda: self._set_progress_bar(mode='indeterminate', start_animation=True))
        except tk.TclError:
            # 忽略UI更新错误
            pass

        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0, encoding='utf-8', errors='replace')
        
        stdout_output = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    stdout_output.append(line)
                    self._log(line.strip()) # Log live output
        
        stderr_output = []
        if process.stderr: # Also capture stderr live if possible, or use communicate
             for line in iter(process.stderr.readline, ''):
                if line:
                    stderr_output.append(line)
                    self._log(line.strip(), is_error=True) # Log live error output

        process.wait() # Ensure process is finished
        # stdout_data, stderr_data = process.communicate() # This would be after live reading

        # 安全地重置进度条
        try:
            if self.app.master.winfo_exists():
                self.app.master.after(0, lambda: self._set_progress_bar(mode='determinate', value=0)) # Reset after step
        except tk.TclError:
            # 忽略UI更新错误
            pass

        # if stdout_data: self._log(stdout_data)
        # if stderr_data: self._log(stderr_data, is_error=True)

        if process.returncode != 0:
            # combined_stderr = "".join(stderr_output) # If reading live
            # raise Exception(f"{step_name} 失败，返回码 {process.returncode}.错误: {combined_stderr or '见日志'}")
            # Check if stderr_output is already logged, if not, log it here from communicate() if preferred
            raise Exception(f"{step_name} 失败，返回码 {process.returncode}. 详情见上方日志。")
        self._log(f"{step_name} 成功完成。")


    def _export_onnx_to_saved_model(self, onnx_path, saved_model_dir):
        """Converts ONNX to TensorFlow SavedModel using onnx-tf."""
        try:
            import onnx_tf # Check if installed
            # onnx-tf convert -i <INPUT_ONNX_FILE> -o <OUTPUT_SAVED_MODEL_DIR>
            cmd = [sys.executable, "-m", "onnx_tf.cli", "convert", "-i", onnx_path, "-o", saved_model_dir]
            self._run_subprocess_command(cmd, "ONNX到SavedModel转换")
            return True
        except ImportError as e:
            self._log(f"onnx-tf库未安装。请运行 'pip install onnx-tf'，详细信息: {e}", True)
            messagebox.showerror("缺少库", f"需要 onnx-tf 库。请手动安装: pip install onnx-tf\n详细信息: {e}", parent=self.app.master)
            return False
        except Exception as e:
            self._log(f"ONNX到SavedModel转换失败: {e}", True)
            messagebox.showerror("ONNX到SavedModel转换失败", f"发生异常: {e}", parent=self.app.master)
            return False


    def _export_saved_model_to_tflite(self, saved_model_dir, tflite_output_path, optimization_str, quantization_type):
        """Converts TensorFlow SavedModel to TFLite."""
        self._log("开始从SavedModel转换到TFLite...")
        try:
            import tensorflow as tf
            # 安全更新UI
            try:
                if self.app.master.winfo_exists():
                    self.app.master.after(0, self._update_progress_label, "正在转换SavedModel到TFLite...")
                    self.app.master.after(0, lambda: self._set_progress_bar(mode='indeterminate', start_animation=True))
            except tk.TclError:
                pass

            # 检查SavedModel目录是否存在和有效
            if not os.path.exists(saved_model_dir):
                self._log(f"SavedModel目录不存在: {saved_model_dir}", is_error=True)
                return False
                
            # 尝试使用更安全的方式加载SavedModel    
            self._log(f"正在从{saved_model_dir}加载SavedModel...")
            try:
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            except Exception as e:
                self._log(f"加载SavedModel失败: {e}", is_error=True)
                self._log("尝试使用直接转换方法...")
                # 这里可以添加其他转换方法，如直接从ONNX转换等
                return False
            
            # Apply optimizations
            try:
                if optimization_str == "OPTIMIZE_FOR_SIZE":
                    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
                elif optimization_str == "OPTIMIZE_FOR_LATENCY":
                    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
                elif optimization_str == "DEFAULT":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # NO_OPTIMIZATION is default if not set

                # Apply quantization (example)
                if quantization_type == "FP16" and hasattr(tf.lite.Optimize, 'DEFAULT'): # FP16 usually with default opts
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    self._log("应用TFLite FP16量化。")
                elif quantization_type == "INT8 (Dynamic)" and hasattr(tf.lite.Optimize, 'DEFAULT'):
                    converter.optimizations = [tf.lite.Optimize.DEFAULT] # INT8 dynamic range often with default
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    self._log("应用TFLite INT8动态范围量化。")
            except Exception as e:
                self._log(f"设置优化参数失败，使用默认配置: {e}", is_error=True)
                
            # 执行转换
            self._log("开始TFLite转换...")
            try:
                tflite_model = converter.convert()
                with open(tflite_output_path, 'wb') as f:
                    f.write(tflite_model)
                self._log(f"成功导出到TFLite: {tflite_output_path}")
                return True
            except Exception as e:
                self._log(f"TFLite转换失败: {e}", is_error=True)
                return False
        except ImportError as e:
            self._log(f"TensorFlow库未安装: {e}", True)
            self._log("请安装TensorFlow: pip install tensorflow")
            return False
        except Exception as e:
            self._log(f"SavedModel到TFLite转换过程中发生未预期错误: {e}", True)
            return False
        finally:
            # 安全重置UI
            try:
                if self.app.master.winfo_exists():
                    self.app.master.after(0, lambda: self._set_progress_bar(mode='determinate', value=0))
            except tk.TclError:
                pass


    def _export_onnx_to_tflite(self, options):
        try:
            onnx_input_path = options["input_path"]
            tflite_output_path = options["output_path"]
            output_dir = options["output_dir"]
            tf_optimization = options.get("tflite_optimization", "DEFAULT")
            quantization = options.get("quantization", "None") # Get general quantization option
            
            # 验证输入文件
            if not os.path.exists(onnx_input_path):
                self._log(f"ONNX输入文件不存在: {onnx_input_path}", is_error=True)
                self._export_finished(False, "ONNX输入文件不存在")
                return
            
            # 检查输出目录
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    self._log(f"创建输出目录: {output_dir}")
                except Exception as e:
                    self._log(f"创建输出目录失败: {e}", is_error=True)
                    self._export_finished(False, f"创建输出目录失败: {e}")
                    return

            # Intermediate SavedModel directory
            saved_model_dir = os.path.join(output_dir, options["output_filename_base"] + "_saved_model")
            if os.path.exists(saved_model_dir):
                import shutil
                try:
                    shutil.rmtree(saved_model_dir) # Clean up previous attempt
                    self._log(f"清理之前的SavedModel目录: {saved_model_dir}")
                except Exception as e:
                    self._log(f"清理SavedModel目录失败: {e}", is_error=True)

            # 执行转换流程
            self._execute_tflite_conversion(onnx_input_path, saved_model_dir, tflite_output_path, tf_optimization, quantization)
            
        except Exception as e:
            self._log(f"ONNX到TFLite导出过程中发生错误: {e}", is_error=True)
            self._export_finished(False, f"导出失败: {e}")
            
    def _execute_tflite_conversion(self, onnx_input_path, saved_model_dir, tflite_output_path, tf_optimization, quantization):
        """执行实际的TFLite转换流程"""
        try:
            # 执行ONNX到SavedModel转换
            self._log("开始ONNX到SavedModel转换...")
            if not self._export_onnx_to_saved_model(onnx_input_path, saved_model_dir):
                # 如果直接转换失败，尝试备用方法
                self._log("标准ONNX到SavedModel转换失败，尝试备用方法...")
                if not self._try_alternative_onnx_conversion(onnx_input_path, tflite_output_path, tf_optimization, quantization):
                    raise Exception("所有ONNX转换方法都失败")
                else:
                    # 备用方法成功，直接完成
                    self._export_finished(True, f"使用备用方法成功导出到 {os.path.basename(tflite_output_path)}")
                    return
            
            # 执行SavedModel到TFLite转换
            self._log("开始SavedModel到TFLite转换...")
            if not self._export_saved_model_to_tflite(saved_model_dir, tflite_output_path, tf_optimization, quantization):
                # 如果转换失败，尝试备用方法
                self._log("标准SavedModel到TFLite转换失败，尝试备用方法...")
                if not self._try_alternative_tflite_conversion(saved_model_dir, tflite_output_path, tf_optimization, quantization):
                    raise Exception("所有TFLite转换方法都失败") 
            
            self._export_finished(True, f"成功导出到 {os.path.basename(tflite_output_path)}")
            
        except Exception as e: # Catch failures from helper methods
            self._log(f"ONNX到TFLite导出链失败: {e}", True)
            self._export_finished(False, f"TFLite导出失败: {e}")
        finally:
            # 安全清理
            if os.path.exists(saved_model_dir):
                try:
                    import shutil
                    shutil.rmtree(saved_model_dir)
                    self._log(f"清理临时SavedModel目录: {saved_model_dir}")
                except Exception as e_rm: 
                    self._log(f"清理SavedModel时出错: {e_rm}", True)

    def _try_alternative_onnx_conversion(self, onnx_input_path, tflite_output_path, optimization_str, quantization_type):
        """尝试备用的ONNX到TFLite转换方法"""
        self._log("尝试使用备用方法进行ONNX到TFLite转换...")
        try:
            # 尝试使用onnx2tf库直接转换
            self._log("检查onnx2tf库是否可用...")
            try:
                import onnx2tf
                self._log("找到onnx2tf库，尝试直接转换...")
                cmd = [sys.executable, "-m", "onnx2tf", "--output_saved_model", "--output_tflite", 
                       "--output_no_nchw", "--input_path", onnx_input_path, 
                       "--output_path", os.path.dirname(tflite_output_path)]
                self._run_subprocess_command(cmd, "onnx2tf直接转换")
                
                # 重命名输出文件为期望的文件名
                expected_tflite_path = os.path.join(
                    os.path.dirname(tflite_output_path),
                    os.path.basename(onnx_input_path).replace('.onnx', '.tflite')
                )
                if os.path.exists(expected_tflite_path) and expected_tflite_path != tflite_output_path:
                    import shutil
                    shutil.move(expected_tflite_path, tflite_output_path)
                
                return True
            except ImportError:
                self._log("onnx2tf库不可用，尝试下一个备用方法...")
                
            # 尝试使用tf2onnx库的转换命令
            self._log("尝试使用tf2onnx库...")
            try:
                # 检查是否安装了tf2onnx
                import tf2onnx
                self._log("找到tf2onnx库，尝试反向转换...")
                # 这里注意：tf2onnx主要用于TF到ONNX，而非ONNX到TF
                self._log("tf2onnx不适用于ONNX到TF转换，跳过此方法")
            except ImportError:
                self._log("tf2onnx库不可用")
            
            # 更多的备用方法可以在这里添加...
            
            return False
        except Exception as e:
            self._log(f"备用转换方法失败: {e}", is_error=True)
            return False
            
    def _try_alternative_tflite_conversion(self, saved_model_dir, tflite_output_path, optimization_str, quantization_type):
        """尝试备用的SavedModel到TFLite转换方法"""
        self._log("尝试使用备用方法进行SavedModel到TFLite转换...")
        try:
            # 尝试使用tflite_runtime
            try:
                self._log("尝试使用tflite_convert命令行工具...")
                cmd = ["tflite_convert", "--saved_model_dir=" + saved_model_dir, 
                       "--output_file=" + tflite_output_path]
                self._run_subprocess_command(cmd, "tflite_convert命令行转换")
                return True
            except Exception as e:
                self._log(f"tflite_convert命令失败: {e}", is_error=True)
            
            # 尝试使用tensorflow的图形模式
            try:
                import tensorflow as tf
                self._log("尝试使用TensorFlow低级API进行转换...")
                # 仅在必要时实现
            except Exception as e:
                self._log(f"TensorFlow低级API转换失败: {e}", is_error=True)
            
            return False
        except Exception as e:
            self._log(f"所有备用TFLite转换方法失败: {e}", is_error=True)
            return False

    def _export_finished(self, success=True, message=""):
        # 使用更安全的方法调用UI更新
        try:
            # 确保主窗口仍然存在
            if self.app.master.winfo_exists():
                self.app.master.after(0, self._safely_update_ui_after_export, success, message)
        except tk.TclError:
            # 主窗口可能已被删除，忽略错误
            pass
    
    def _safely_update_ui_after_export(self, success, message):
        """安全地更新UI状态，处理可能的Tcl/Tk错误"""
        try:
            # 恢复按钮状态
            if hasattr(self.app, 'start_export_btn') and self.app.start_export_btn.winfo_exists():
                self.app.start_export_btn.config(state="normal")
            
            # 更新标签和进度条
            if success:
                self._update_progress_label(f"完成: {message}")
                self._set_progress_bar(mode='determinate', value=100)
            else:
                self._update_progress_label(f"失败: {message}")
                self._set_progress_bar(mode='determinate', value=0)
            
            # 延迟3秒后重置UI
            if self.app.master.winfo_exists():
                self.app.master.after(3000, lambda: self._update_progress_label("选择模型并导出"))
                self.app.master.after(3000, lambda: self._set_progress_bar(mode='determinate', value=0))
        except tk.TclError as e:
            # 捕获所有Tcl/Tk错误
            print(f"导出完成后更新UI时出错: {e}")
            # 可选：记录到日志文件但不尝试更新UI

    def _export_onnx_to_openvino(self, options):
        """将ONNX模型转换为OpenVINO IR格式"""
        try:
            # Check for OpenVINO Model Optimizer ('mo' or 'mo.py')
            mo_command = "mo" # Assume 'mo' is in PATH
            
            # 尝试使用系统环境变量中的OpenVINO安装路径
            openvino_bin = os.environ.get("INTEL_OPENVINO_DIR")
            if openvino_bin:
                mo_path = os.path.join(openvino_bin, "deployment_tools", "model_optimizer", "mo.py")
                if os.path.exists(mo_path):
                    mo_command = [sys.executable, mo_path]
                    self._log(f"找到OpenVINO Model Optimizer: {mo_path}")
                else:
                    # 尝试找到新版本OpenVINO的mo路径
                    mo_path = os.path.join(openvino_bin, "tools", "mo", "mo.py")
                    if os.path.exists(mo_path):
                        mo_command = [sys.executable, mo_path]
                        self._log(f"找到OpenVINO Model Optimizer: {mo_path}")

            # 构建命令行参数
            cmd = []
            if isinstance(mo_command, list):
                cmd.extend(mo_command)
            else:
                cmd.append(mo_command)
                
            cmd.extend(["--input_model", options["input_path"], "--output_dir", options["output_dir"]])
            cmd.extend(["--model_name", options["output_filename_base"]]) # This sets .xml/.bin base name
            
            precision = options.get("openvino_precision", "FP32")
            if precision != "FP32": # FP32 is often default, FP16 needs to be specified
                 cmd.extend(["--data_type", precision])
            
            input_shape = options.get("openvino_input_shape")
            if input_shape:
                cmd.extend(["--input_shape", str(input_shape)]) # Ensure it's a string format MO expects
            
            # 如果未提供输入形状，使用默认值
            if not input_shape:
                # 使用默认值或从ONNX模型中推断
                self._log("未指定OpenVINO输入形状，将从模型推断或使用默认值")

            # Handle quantization
            quant_type = options.get("quantization", "None")
            if quant_type != "None":
                self._log(f"OpenVINO量化 ({quant_type}) 通常通过NNCF或POT工具进行，比命令行参数更复杂。此导出将尝试基于precision参数。", is_error=True)

            # 安全运行命令
            try:
                self._run_subprocess_command(cmd, "ONNX到OpenVINO IR转换")
                self._export_finished(True, f"成功导出OpenVINO IR到 {options['output_dir']}")
            except Exception as e:
                # 如果命令失败，尝试使用openvino库直接转换
                self._log(f"命令行转换失败，尝试使用OpenVINO API: {e}", is_error=True)
                self._try_openvino_api_conversion(options)

        except FileNotFoundError: # If 'mo' command is not found
            self._log("OpenVINO Model Optimizer (mo) 未找到。尝试使用OpenVINO API进行转换...", True)
            self._try_openvino_api_conversion(options)
        except Exception as e:
            self._log(f"ONNX到OpenVINO IR转换失败: {e}", True)
            self._export_finished(False, f"OpenVINO导出失败: {e}")
            
    def _try_openvino_api_conversion(self, options):
        """尝试使用OpenVINO API进行模型转换"""
        try:
            self._log("尝试使用OpenVINO API进行模型转换...")
            try:
                import openvino as ov
                self._log("找到OpenVINO API")
                
                # 从ONNX文件读取模型
                self._log(f"正在从{options['input_path']}读取ONNX模型...")
                model = ov.Core().read_model(options['input_path'])
                
                # 编译和保存模型
                output_path = os.path.join(options['output_dir'], f"{options['output_filename_base']}.xml")
                self._log(f"正在保存IR模型到{output_path}...")
                
                # 只保存，不传递精度参数
                ov.serialize(model, output_path)
                
                self._log(f"OpenVINO API转换成功: {output_path}")
                self._export_finished(True, f"成功导出OpenVINO IR到 {options['output_dir']}")
                return True
            except ImportError:
                self._log("找不到OpenVINO API，请安装: pip install openvino", is_error=True)
                self._export_finished(False, "OpenVINO API未安装")
                return False
        except Exception as e:
            self._log(f"OpenVINO API转换失败: {e}", is_error=True)
            self._export_finished(False, f"OpenVINO API导出失败: {e}")
            return False

    def update_ui_for_license(self):
        # 这里可以根据实际需要刷新导出页面的UI
        pass