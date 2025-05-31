# training_handler.py
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog # Ensure simpledialog is imported if used
import os
import sys
import subprocess
import threading
import platform
import re
import shutil
from urllib.request import urlretrieve
import zipfile
from config import config # 直接从config导入config

# Import security module
try:
    from security import protect_training_feature, limit_training_images, check_feature_access, get_license_info
except ImportError:
    # If security module doesn't exist, use empty decorator
    def protect_training_feature(feature_name):
        def decorator(func): return func
        return decorator
    def limit_training_images(*args, **kwargs): return True

class TrainingHandler:
    def __init__(self, app):
        self.app = app
        self.training_process = None
        self.is_training = False 
        self.stop_training_flag = threading.Event()
        self.weights_dir = os.path.join(os.getcwd(), "pretrained_weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.cuda_available = False
        self.cuda_version = None
        self.checkpoint_dir = os.path.join(os.getcwd(), "training_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.last_checkpoint_path = None

    def _set_progress_indeterminate(self, start_animation=True):
        """Helper to set progress bar to indeterminate mode."""
        if hasattr(self.app, 'progress_bar') and self.app.progress_bar:
            self.app.progress_bar.config(mode='indeterminate')
            if start_animation:
                self.app.progress_bar.start(10) # Start animation, 10ms interval

    def _set_progress_determinate(self, value=0):
        """Helper to set progress bar to determinate mode and stop animation."""
        if hasattr(self.app, 'progress_bar') and self.app.progress_bar:
            self.app.progress_bar.stop()
            self.app.progress_bar.config(mode='determinate', value=value)

    def _update_progress_label(self, text):
        """Helper to update the progress label."""
        if hasattr(self.app, 'progress_label') and self.app.progress_label:
            self.app.progress_label.config(text=text)


    def _detect_cuda_version(self):
        # ... (no changes needed here for progress bar)
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
            if result.returncode == 0:
                output = result.stdout
                match = re.search(r'CUDA Version: (\d+\.\d+)', output)
                if match: return match.group(1)
        except Exception: pass
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
            if result.returncode == 0:
                output = result.stdout
                match = re.search(r'release (\d+\.\d+)', output)
                if match: return match.group(1)
        except Exception: pass
        return None

    def detect_system_capabilities(self):
        # ... (no changes needed here for progress bar, it's quick)
        self.app._add_to_output_log_queue("正在检测系统能力...\n", is_train=True)
        try:
            import torch
            if torch.cuda.is_available():
                self.cuda_available = True
                self.cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                self.app._add_to_output_log_queue(f"✓ CUDA {self.cuda_version} 可用\n", is_train=True)
                self.app._add_to_output_log_queue(f"✓ 检测到 {gpu_count} 个GPU: {gpu_name}\n", is_train=True)
                self.app.device_var.set("0") 
            else:
                self.app._add_to_output_log_queue("⚠ CUDA不可用，将使用CPU训练\n", is_train=True)
                self.app.device_var.set("cpu")
        except ImportError:
            self.app._add_to_output_log_queue("⚠ PyTorch未安装。请点击'检测/安装 PyTorch'按钮。\n", is_train=True)
            self.app.device_var.set("cpu")
        except Exception as e:
            self.app._add_to_output_log_queue(f"⚠ 检测CUDA时出错: {e}\n", is_train=True)
            self.app.device_var.set("cpu")
        
        if hasattr(self.app, 'env_status_label'):
             status_text = f"CUDA: {'可用 ' + str(self.cuda_version) if self.cuda_available else '不可用'}"
             try:
                 import torch
                 status_text += f" | PyTorch: {torch.__version__}"
             except ImportError:
                 status_text += " | PyTorch: 未安装"
             self.app.env_status_label.config(text=status_text)


    def fix_environment_issues(self):
        def fix_thread():
            original_progress_label_text = self.app.progress_label.cget("text") # Save original text
            try:
                self.app._add_to_output_log_queue("开始修复环境兼容性问题...\n")
                self.app.master.after(0, lambda: self.app.env_status_label.config(text="正在修复环境问题..."))
                self.app.master.after(0, self._update_progress_label, "正在卸载软件包...")
                self.app.master.after(0, self._set_progress_determinate, 0) # Reset first
                
                uninstall_packages = ["numpy", "opencv-python", "opencv-contrib-python", "opencv-python-headless"]
                self.app._add_to_output_log_queue("步骤1: 卸载冲突的包...\n")
                for i, package in enumerate(uninstall_packages):
                    self.app._add_to_output_log_queue(f"卸载 {package}...\n")
                    # self.app.master.after(0, self._update_progress_label, f"正在卸载 {package}...") # Too fast to show
                    uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", package, "-y"]
                    subprocess.run(uninstall_cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
                self.app.master.after(0, self._set_progress_determinate, 10)


                self.app.master.after(0, self._update_progress_label, "正在清理pip缓存...")
                self.app._add_to_output_log_queue("步骤2: 清理pip缓存...\n")
                cache_cmd = [sys.executable, "-m", "pip", "cache", "purge"]
                subprocess.run(cache_cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
                self.app.master.after(0, self._set_progress_determinate, 20)
                
                # Install NumPy
                self.app.master.after(0, self._update_progress_label, "正在安装 NumPy (可能需要几分钟)...")
                self.app.master.after(0, self._set_progress_indeterminate)
                self.app._add_to_output_log_queue("步骤3: 安装兼容版本的NumPy...\n")
                numpy_cmd = [sys.executable, "-m", "pip", "install", "numpy==1.24.3", "--no-cache-dir"]
                numpy_process = subprocess.Popen(numpy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0, encoding='utf-8', errors='replace')
                numpy_stdout, numpy_stderr = numpy_process.communicate()
                self.app.master.after(0, self._set_progress_determinate, 50) # Assume 50% after numpy

                if numpy_process.returncode == 0:
                    self.app._add_to_output_log_queue("NumPy 1.24.3 安装成功\n")
                else:
                    self.app._add_to_output_log_queue(f"NumPy 安装失败: {numpy_stderr}\n", is_error=True)
                    self.app.master.after(0, lambda: self.app.env_status_label.config(text="环境修复失败 (NumPy)"))
                    self.app.master.after(0, self._update_progress_label, "NumPy 安装失败")
                    return
                
                # Install OpenCV
                self.app.master.after(0, self._update_progress_label, "正在安装 OpenCV (可能需要几分钟)...")
                self.app.master.after(0, self._set_progress_indeterminate)
                self.app._add_to_output_log_queue("步骤4: 安装兼容的OpenCV...\n")
                cv_cmd = [sys.executable, "-m", "pip", "install", "opencv-python==4.8.1.78", "--no-cache-dir"]
                cv_process = subprocess.Popen(cv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0, encoding='utf-8', errors='replace')
                cv_stdout, cv_stderr = cv_process.communicate()
                self.app.master.after(0, self._set_progress_determinate, 90) # Assume 90% after opencv

                if cv_process.returncode == 0:
                    self.app._add_to_output_log_queue("OpenCV 4.8.1.78 安装成功\n")
                else:
                    self.app._add_to_output_log_queue(f"OpenCV 安装失败: {cv_stderr}\n", is_error=True)
                    self.app.master.after(0, lambda: self.app.env_status_label.config(text="环境修复失败 (OpenCV)"))
                    self.app.master.after(0, self._update_progress_label, "OpenCV 安装失败")
                    return
                
                self.app.master.after(0, self._update_progress_label, "正在验证安装...")
                self.app._add_to_output_log_queue("步骤5: 验证安装...\n")
                try:
                    import importlib, numpy as np, cv2 # Need to import them here for the thread
                    modules_to_reload = ['numpy', 'cv2']
                    for module_name in modules_to_reload:
                        if module_name in sys.modules:
                            importlib.reload(sys.modules[module_name]) # Reload in current thread
                    
                    # Re-import after reload to ensure the new versions are used in this thread's scope
                    np_version = importlib.import_module('numpy').__version__
                    cv2_version = importlib.import_module('cv2').__version__

                    self.app._add_to_output_log_queue(f"✓ NumPy版本: {np_version}\n")
                    self.app._add_to_output_log_queue(f"✓ OpenCV版本: {cv2_version}\n")
                    
                    test_img = np.zeros((100, 100, 3), dtype=np.uint8) # np from this thread
                    cv2.resize(test_img, (50, 50)) # cv2 from this thread
                    self.app._add_to_output_log_queue("✓ OpenCV功能测试通过\n")
                    
                    self.app._add_to_output_log_queue("环境修复成功！\n")
                    self.app.master.after(0, lambda: self.app.env_status_label.config(text="环境已修复"))
                    self.app.master.after(0, self._update_progress_label, "环境修复成功！")
                    self.app.master.after(0, self._set_progress_determinate, 100)
                    self.app.master.after(0, lambda: messagebox.showinfo("修复成功", "环境问题已修复！", parent=self.app.master))
                except ImportError as e:
                    self.app._add_to_output_log_queue(f"验证失败: {e}\n", is_error=True)
                    self.app.master.after(0, lambda: self.app.env_status_label.config(text="环境修复失败 (验证)"))
                    self.app.master.after(0, self._update_progress_label, "修复后验证失败")
                except Exception as e: 
                    self.app._add_to_output_log_queue(f"验证时发生错误: {e}\n", is_error=True)
                    self.app.master.after(0, lambda: self.app.env_status_label.config(text="环境修复失败 (验证错误)"))
                    self.app.master.after(0, self._update_progress_label, "修复后验证出错")

            except Exception as e:
                error_msg = f"修复环境时出错: {e}"
                self.app._add_to_output_log_queue(error_msg + "\n", is_error=True)
                self.app.master.after(0, lambda: self.app.env_status_label.config(text="环境修复出错"))
                self.app.master.after(0, self._update_progress_label, "环境修复出错")
            finally:
                # Reset progress bar and label after a delay or if an error occurred early
                self.app.master.after(2000, lambda: self._set_progress_determinate(0))
                self.app.master.after(2000, lambda: self._update_progress_label(original_progress_label_text))


        if messagebox.askyesno("修复环境", 
                               "这将重新安装NumPy和OpenCV包到兼容版本。\n"
                               "可能需要几分钟时间，是否继续？", 
                               parent=self.app.master):
            threading.Thread(target=fix_thread, daemon=True).start()

    def install_pytorch_cuda(self):
        def install_thread():
            original_progress_label_text = self.app.progress_label.cget("text")
            try:
                self.app._add_to_output_log_queue("开始检测和安装PyTorch环境...\n")
                self.app.master.after(0, lambda: self.app.env_status_label.config(text="正在检测和安装PyTorch..."))
                self.app.master.after(0, self._update_progress_label, "正在准备 PyTorch 安装...")
                self.app.master.after(0, self._set_progress_determinate, 0) # Reset
                
                cuda_version_detected = self._detect_cuda_version()
                torch_package_name_for_log = "PyTorch" # Generic name for logging

                if cuda_version_detected:
                    self.app._add_to_output_log_queue(f"检测到CUDA版本: {cuda_version_detected}\n")
                    if cuda_version_detected.startswith("12"):
                        torch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
                        torch_package_name_for_log = "PyTorch (CUDA 12.1)"
                    elif cuda_version_detected.startswith("11"):
                        torch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
                        torch_package_name_for_log = "PyTorch (CUDA 11.8)"
                    else: 
                        self.app._add_to_output_log_queue("检测到未知或较旧的CUDA版本，尝试通用CUDA PyTorch安装。\n")
                        torch_package = "torch torchvision torchaudio" 
                        torch_package_name_for_log = "PyTorch (CUDA)"
                else:
                    self.app._add_to_output_log_queue("未检测到CUDA，将安装CPU版本的PyTorch\n")
                    torch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                    torch_package_name_for_log = "PyTorch (CPU)"
                
                self.app.master.after(0, self._update_progress_label, f"正在安装 {torch_package_name_for_log} (可能需要较长时间)...")
                self.app.master.after(0, self._set_progress_indeterminate) # Indeterminate for the whole pip command
                self.app._add_to_output_log_queue(f"执行安装: {torch_package}...\n")
                install_cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + torch_package.split()
                
                process = subprocess.Popen(
                    install_cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
                    encoding='utf-8', errors='replace'
                )
                # Stream output to log, but don't try to parse for progress here for simplicity
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        if line: self.app._add_to_output_log_queue(line)
                stdout_data, stderr_data = process.communicate() # Get remaining output and wait
                if stdout_data: self.app._add_to_output_log_queue(stdout_data)


                self.app.master.after(0, self._set_progress_determinate, 100) # Mark as 100% after command finishes
                
                if process.returncode == 0:
                    self.app._add_to_output_log_queue("PyTorch安装成功！\n")
                    self.app.master.after(0, self._update_progress_label, f"{torch_package_name_for_log} 安装成功!")
                    self.app.master.after(0, self.detect_system_capabilities) 
                    self.app.master.after(0, lambda: self.app.env_status_label.config(text="PyTorch环境已就绪"))
                else:
                    error_msg = f"PyTorch安装失败: {stderr_data}"
                    self.app._add_to_output_log_queue(error_msg + "\n", is_error=True)
                    self.app.master.after(0, self._update_progress_label, f"{torch_package_name_for_log} 安装失败.")
                    self.app.master.after(0, lambda: self.app.env_status_label.config(text="PyTorch环境安装失败"))
            except Exception as e:
                error_msg = f"PyTorch安装过程中出现错误: {e}"
                self.app._add_to_output_log_queue(error_msg + "\n", is_error=True)
                self.app.master.after(0, self._update_progress_label, "PyTorch 安装出错.")
                self.app.master.after(0, lambda: self.app.env_status_label.config(text="PyTorch环境安装出错"))
            finally:
                self.app.master.after(2000, lambda: self._set_progress_determinate(0))
                self.app.master.after(2000, lambda: self._update_progress_label(original_progress_label_text))

        threading.Thread(target=install_thread, daemon=True).start()

    # ... (select_datasets_yaml, on_yolo_version_change, etc. remain the same)
    # ... (download_pretrained_weights, _download_weight_file also remain mostly same,
    #      _download_weight_file already has determinate progress for file download)

    # Ensure _update_weights_combo is still here and correct
    def _update_weights_combo(self):
        # ... (this method should be fine as it was)
        weights_files = []
        if os.path.exists(self.weights_dir):
            for file in os.listdir(self.weights_dir):
                if file.endswith('.pt'):
                    weights_files.append(os.path.join(self.weights_dir, file)) 
        
        framework = self.app.selected_yolo_version_name.get()
        current_selection = self.app.weights_var.get()
        
        if framework in config.YOLO_VERSIONS:
            default_weights = config.YOLO_VERSIONS[framework].get("default_weights", {})
            for weight_name in default_weights.keys():
                full_path_candidate = os.path.abspath(os.path.join(self.weights_dir, weight_name))
                if not any(os.path.abspath(f) == full_path_candidate for f in weights_files):
                    if weight_name != current_selection or not current_selection.endswith(".pt"):
                        if weight_name not in [os.path.basename(f) for f in weights_files]:
                             weights_files.append(weight_name) 

        processed_weights_files = []
        seen_basenames_or_fullpath = set()

        for f_item in weights_files:
            if os.path.exists(f_item) and os.path.isfile(f_item): 
                abs_path = os.path.abspath(f_item)
                if abs_path not in seen_basenames_or_fullpath:
                    processed_weights_files.append(abs_path)
                    seen_basenames_or_fullpath.add(abs_path)
                    seen_basenames_or_fullpath.add(os.path.basename(abs_path)) 
            elif not os.path.isabs(f_item) and f_item.endswith('.pt'): 
                if f_item not in seen_basenames_or_fullpath:
                    processed_weights_files.append(f_item)
                    seen_basenames_or_fullpath.add(f_item)
        
        if current_selection and os.path.exists(current_selection) and os.path.isfile(current_selection):
            abs_current_selection = os.path.abspath(current_selection)
            if abs_current_selection not in processed_weights_files:
                processed_weights_files.append(abs_current_selection)
        elif current_selection and current_selection.endswith('.pt') and not os.path.isabs(current_selection):
             if current_selection not in processed_weights_files: 
                  processed_weights_files.append(current_selection)

        self.app.weights_combo['values'] = sorted(list(set(processed_weights_files))) 

        if current_selection and current_selection in self.app.weights_combo['values']:
            self.app.weights_var.set(current_selection)
        elif not current_selection and self.app.weights_combo['values']:
            pass 
        elif current_selection and current_selection not in self.app.weights_combo['values']:
             if self.app.weights_combo['values']:
                if framework in config.YOLO_VERSIONS:
                    default_weights_map = config.YOLO_VERSIONS[framework].get("default_weights", {})
                    if default_weights_map:
                        framework_prefix = framework.lower()
                        preferred_default = f"{framework_prefix}n.pt" if f"{framework_prefix}n.pt" in default_weights_map else list(default_weights_map.keys())[0]
                        if preferred_default in self.app.weights_combo['values']:
                            self.app.weights_var.set(preferred_default)
                        else: 
                             self.app.weights_var.set(self.app.weights_combo['values'][0])
                    elif self.app.weights_combo['values']:
                        self.app.weights_var.set(self.app.weights_combo['values'][0]) 
                    else: self.app.weights_var.set("")
                elif self.app.weights_combo['values']:
                     self.app.weights_var.set(self.app.weights_combo['values'][0]) 
                else: self.app.weights_var.set("")
             else: self.app.weights_var.set("")
    
    # ... (setup_yolo_code, _download_yolo_code_threaded, etc.)
    # The _install_requirements_threaded method already uses indeterminate progress correctly.
    # So, we only needed to update fix_environment_issues and install_pytorch_cuda.

    # Ensure all other methods like setup_yolo_code, start_training, etc., are still present
    # I will only show the changed ones and assume others are unchanged from previous version.

    # --- Placeholder for other methods from your `training_handler.py` ---
    def select_datasets_yaml(self):
        # (Original implementation)
        file_path = filedialog.askopenfilename(
            title="选择 datasets.yaml 文件",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            parent=self.app.master
        )
        if file_path:
            self.app.datasets_yaml_path.set(file_path)
            self.app.status_bar.config(text=f"数据集: {os.path.basename(file_path)}")

    def on_yolo_version_change(self, event=None):
        # (Original implementation)
        selected_framework = self.app.selected_yolo_version_name.get()
        self.app.selected_yolo_subversion.set("")
        if selected_framework in config.YOLO_VERSIONS:
            versions = config.YOLO_VERSIONS[selected_framework]["versions"]
            self.app.yolo_subversion_combo.configure(values=versions, state="readonly")
            if versions:
                self.app.yolo_subversion_combo.set(versions[0])
        self.app.yolo_code_path.set("未设置YOLO代码路径")
        current_weight = self.app.weights_var.get()
        if selected_framework in config.YOLO_VERSIONS:
            default_weights_map = config.YOLO_VERSIONS[selected_framework].get("default_weights", {})
            if default_weights_map:
                preferred_default = None
                framework_prefix = selected_framework.lower()
                if f"{framework_prefix}n.pt" in default_weights_map:
                    preferred_default = f"{framework_prefix}n.pt"
                elif default_weights_map: 
                    preferred_default = list(default_weights_map.keys())[0]
                if preferred_default:
                    self.app.weights_var.set(preferred_default)
        self._update_weights_combo()
        if self.app.selected_yolo_subversion.get():
             self.on_yolo_subversion_change()

    def on_yolo_subversion_change(self, event=None):
        # (Original implementation)
        self._update_weights_combo()

    def browse_weights_file(self):
        # (Original implementation)
        file_path = filedialog.askopenfilename(
            title="选择预训练权重文件",
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")],
            parent=self.app.master
        )
        if file_path:
            self.app.weights_var.set(os.path.abspath(file_path))
            self._update_weights_combo()

    def browse_project_dir(self):
        # (Original implementation)
        dir_path = filedialog.askdirectory(
            title="选择训练输出文件夹",
            parent=self.app.master
        )
        if dir_path:
            self.app.project_dir_var.set(dir_path)
            
    def download_pretrained_weights(self):
        # (Original implementation - it calls _download_weight_file which has progress)
        framework = self.app.selected_yolo_version_name.get()
        if not framework or framework not in config.YOLO_VERSIONS:
            messagebox.showwarning("未选择框架", "请先选择YOLO框架版本", parent=self.app.master)
            return
        weights_info = config.YOLO_VERSIONS[framework].get("default_weights", {})
        if not weights_info:
            messagebox.showinfo("无可用权重", "该框架版本没有预定义的权重下载链接", parent=self.app.master)
            return
        weights_dialog = tk.Toplevel(self.app.master)
        weights_dialog.title("选择预训练权重")
        weights_dialog.geometry("400x300")
        weights_dialog.configure(bg=config.UI_BACKGROUND_COLOR)
        weights_dialog.transient(self.app.master)
        weights_dialog.grab_set()
        tk.Label(weights_dialog, text="选择要下载的预训练权重:", background=config.UI_BACKGROUND_COLOR, foreground=config.UI_FOREGROUND_COLOR).pack(pady=10)
        weights_listbox = tk.Listbox(weights_dialog, height=8, bg=config.UI_BACKGROUND_COLOR, fg=config.UI_FOREGROUND_COLOR, 
                                     selectbackground=config.UI_SELECT_BACKGROUND_COLOR, selectforeground=config.UI_FOREGROUND_COLOR)
        weights_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        for weight_name in weights_info.keys():
            weights_listbox.insert(tk.END, weight_name)
        def download_selected():
            selection = weights_listbox.curselection()
            if not selection:
                messagebox.showwarning("未选择", "请选择要下载的权重", parent=weights_dialog)
                return
            weight_name = weights_listbox.get(selection[0])
            weight_url = weights_info[weight_name]
            weights_dialog.destroy()
            self._download_weight_file(weight_name, weight_url) # This already has progress
        button_frame = ttk.Frame(weights_dialog, style="TFrame") # Use ttk.Frame for styling
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="下载", command=download_selected, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=weights_dialog.destroy, style="TButton").pack(side=tk.LEFT, padx=5)

    def _download_weight_file(self, weight_name, weight_url):
        # (Original implementation - this one already has determinate progress)
        def download_thread():
            weight_path = os.path.join(self.weights_dir, weight_name)
            original_progress_label_text = self.app.progress_label.cget("text")
            try:
                if os.path.exists(weight_path):
                    self.app._add_to_output_log_queue(f"权重文件已存在: {weight_name}\n")
                    self.app.weights_var.set(weight_path) 
                    self.app.master.after(0, self._update_weights_combo)
                    return
                
                self.app._add_to_output_log_queue(f"开始下载权重: {weight_name} from {weight_url}\n")
                self.app.master.after(0, self._update_progress_label, f"下载中: {weight_name}")
                self.app.master.after(0, self._set_progress_determinate, 0)

                try:
                    import requests 
                    response = requests.get(weight_url, stream=True)
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    with open(weight_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if total_size > 0:
                                    progress = (downloaded_size / total_size) * 100
                                    self.app.master.after(0, lambda p=progress: self.app.progress_bar.config(value=p))
                except ImportError: 
                    def reporthook(blocknum, blocksize, totalsize):
                        readsofar = blocknum * blocksize
                        if totalsize > 0:
                            percent = readsofar * 1e2 / totalsize
                            self.app.master.after(0, lambda p=percent: self.app.progress_bar.config(value=p))
                    urlretrieve(weight_url, weight_path, reporthook=reporthook)

                self.app._add_to_output_log_queue(f"权重下载完成: {weight_name}\n")
                self.app.weights_var.set(weight_path) 
                self.app.master.after(0, self._update_weights_combo)
                self.app.master.after(0, self._update_progress_label, "权重下载完成")
                self.app.master.after(0, self._set_progress_determinate, 100)
            except Exception as e:
                error_msg = f"下载权重失败 ({weight_name}): {e}"
                self.app._add_to_output_log_queue(error_msg + "\n", is_error=True)
                self.app.master.after(0, self._update_progress_label, "权重下载失败")
                if os.path.exists(weight_path): os.remove(weight_path)
            finally:
                 self.app.master.after(2000, lambda: self._set_progress_determinate(0))
                 self.app.master.after(2000, lambda: self._update_progress_label(original_progress_label_text))
        threading.Thread(target=download_thread, daemon=True).start()


    def setup_yolo_code(self):
        # (Original implementation - it calls _download_yolo_code_threaded which has progress)
        framework_name = self.app.selected_yolo_version_name.get()
        subversion = self.app.selected_yolo_subversion.get()
        if not framework_name or not subversion:
            messagebox.showwarning("版本未选择", "请先选择YOLO框架和具体版本。", parent=self.app.master)
            return
        choice = messagebox.askyesnocancel(
            "设置YOLO代码",
            f"为 {framework_name} {subversion} 设置代码:\n"
            f"  [是] - 下载新代码 (推荐)\n"
            f"  [否] - 选择现有本地文件夹\n"
            f"  [取消] - 返回",
            parent=self.app.master
        )
        if choice is None: return
        elif choice: self._download_yolo_code_threaded(framework_name, subversion)
        else: self._select_existing_yolo_code_threaded(framework_name, subversion)

    def _download_yolo_code_threaded(self, framework_name, subversion_tag):
        # (This already uses reporthook for download progress, and _install_requirements_threaded for indeterminate)
        # No changes needed here for progress if _install_requirements_threaded is correctly using indeterminate.
        base_downloads_dir = os.path.join(os.getcwd(), "yolo_frameworks")
        os.makedirs(base_downloads_dir, exist_ok=True)
        safe_subversion_tag = subversion_tag.replace('.', '_').replace('/', '_')
        target_code_root = os.path.join(base_downloads_dir, f"{framework_name}-{safe_subversion_tag}")
        original_progress_label_text = self.app.progress_label.cget("text")

        def download_and_extract():
            zip_path_local = "" # Define to ensure it's available in finally
            try:
                repo_info = config.YOLO_VERSIONS[framework_name]
                version_for_url = subversion_tag
                if framework_name == "YOLOv5" and not version_for_url.startswith('v'):
                    version_for_url = 'v' + version_for_url
                download_url = repo_info["repo_url_template"].format(version=version_for_url)
                zip_filename = f"{framework_name}-{safe_subversion_tag}.zip"
                zip_path_local = os.path.join(base_downloads_dir, zip_filename) # Assign to local

                self.app._add_to_output_log_queue(f"开始下载 {framework_name} {subversion_tag} from {download_url}...\n")
                self.app.master.after(0, self._update_progress_label, f"下载中: {zip_filename}")
                self.app.master.after(0, self._set_progress_determinate, 0) # For download part

                def reporthook(blocknum, blocksize, totalsize):
                    readsofar = blocknum * blocksize
                    if totalsize > 0:
                        percent = readsofar * 1e2 / totalsize
                        self.app.master.after(0, lambda p=percent: self.app.progress_bar.config(value=p))
                        if readsofar >= totalsize :
                             self.app.master.after(0, self._update_progress_label, f"下载完成: {zip_filename}")
                    else: 
                        self.app.master.after(0, self._update_progress_label, f"下载中: {readsofar/1024:.0f} KB")
                urlretrieve(download_url, zip_path_local, reporthook=reporthook)
                
                self.app._add_to_output_log_queue("下载完成，正在解压...\n")
                self.app.master.after(0, self._update_progress_label, "解压中...")
                self.app.master.after(0, self._set_progress_indeterminate) # Indeterminate for extraction

                if os.path.exists(target_code_root): shutil.rmtree(target_code_root)
                os.makedirs(target_code_root, exist_ok=True)
                with zipfile.ZipFile(zip_path_local, 'r') as zip_ref:
                    # ... (extraction logic as before)
                    members = zip_ref.namelist()
                    if not members: raise Exception("ZIP文件为空")
                    first_member_parts = members[0].replace('\\', '/').split('/')
                    is_single_dir_zip = True
                    top_level_dir_in_zip = ""
                    if len(first_member_parts) > 1 and first_member_parts[0] != "": 
                        top_level_dir_in_zip = first_member_parts[0]
                        for member in members:
                            if not member.replace('\\', '/').startswith(top_level_dir_in_zip + '/'):
                                is_single_dir_zip = False; break
                    else: is_single_dir_zip = False
                    if is_single_dir_zip and top_level_dir_in_zip:
                        temp_extract_path = os.path.join(base_downloads_dir, "_temp_extract")
                        if os.path.exists(temp_extract_path): shutil.rmtree(temp_extract_path)
                        os.makedirs(temp_extract_path)
                        zip_ref.extractall(temp_extract_path)
                        source_dir_to_move = os.path.join(temp_extract_path, top_level_dir_in_zip)
                        for item in os.listdir(source_dir_to_move):
                            s = os.path.join(source_dir_to_move, item)
                            d = os.path.join(target_code_root, item)
                            if os.path.isdir(s): shutil.move(s, d)
                            else: shutil.move(s, d)
                        shutil.rmtree(temp_extract_path)
                    else: zip_ref.extractall(target_code_root)
                
                os.remove(zip_path_local)
                self.app.master.after(0, self._set_progress_determinate, 0) # Reset after extraction
                
                self.app.yolo_code_path.set(os.path.abspath(target_code_root))
                self.app._add_to_output_log_queue(f"代码解压至: {self.app.yolo_code_path.get()}\n")
                self.app.master.after(0, self._update_progress_label, "YOLO代码准备就绪")
                self._install_requirements_threaded(self.app.yolo_code_path.get(), repo_info["requirements"])
            except Exception as e:
                error_msg = f"下载或解压 {framework_name} {subversion_tag} 失败: {e}"
                self.app._add_to_output_log_queue(error_msg + "\n", is_error=True)
                self.app.master.after(0, self._update_progress_label, "YOLO代码下载失败")
                if zip_path_local and os.path.exists(zip_path_local): os.remove(zip_path_local)
                if os.path.exists(target_code_root) and not os.listdir(target_code_root): 
                    os.rmdir(target_code_root)
            finally:
                self.app.master.after(0, self._set_progress_determinate, 0)
                self.app.master.after(2000, lambda: self._update_progress_label(original_progress_label_text))

        threading.Thread(target=download_and_extract, daemon=True).start()


    def _select_existing_yolo_code_threaded(self, framework_name, subversion_tag):
        folder_path = filedialog.askdirectory(
            title=f"选择 {framework_name} {subversion_tag} 代码根文件夹",
            parent=self.app.master
        )
        if folder_path:
            abs_path = os.path.abspath(folder_path)
            if hasattr(self.app, 'set_global_yolo_code_path'):
                self.app.set_global_yolo_code_path(abs_path)
            else:
                self.app.yolo_code_path.set(abs_path)
            self.app._add_to_output_log_queue(f"用户选择YOLO代码路径: {abs_path}\n")
        repo_info = config.YOLO_VERSIONS.get(framework_name, {})
        req_file = repo_info.get("requirements")
        if req_file:
            self._install_requirements_threaded(self.app.yolo_code_path.get(), req_file) # This has progress

    def _install_requirements_threaded(self, code_path, req_file_name):
        # (This method already correctly uses indeterminate progress bar)
        potential_paths = [
            os.path.join(code_path, req_file_name),
            os.path.join(code_path, "ultralytics", req_file_name) 
        ]
        requirements_path = None
        for p_path in potential_paths:
            if os.path.exists(p_path):
                requirements_path = p_path
                break
        if not requirements_path:
            self.app._add_to_output_log_queue(f"注意: 未找到依赖文件 '{req_file_name}' in expected locations within {code_path}.\n")
            return

        original_progress_label_text = self.app.progress_label.cget("text")
        def install():
            try:
                self.app._add_to_output_log_queue(f"开始安装依赖从 {requirements_path}...\n")
                self.app.master.after(0, self._update_progress_label, f"安装依赖中 (从 {os.path.basename(requirements_path)})...")
                self.app.master.after(0, self._set_progress_indeterminate)

                command = [sys.executable, "-m", "pip", "install", "-r", requirements_path, "--upgrade"]
                creationflags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=creationflags, encoding='utf-8', errors='replace')
                
                # Stream stdout for logging
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        if line: self.app._add_to_output_log_queue(line)
                stdout_data, stderr_data = process.communicate() # Get remaining and wait
                if stdout_data: self.app._add_to_output_log_queue(stdout_data)
                
                self.app.master.after(0, self._set_progress_determinate, 0) # Reset after completion

                if process.returncode == 0:
                    self.app._add_to_output_log_queue("依赖安装成功\n")
                    self.app.master.after(0, self._update_progress_label, "依赖安装完成")
                else:
                    self.app._add_to_output_log_queue(f"依赖安装失败 (返回码 {process.returncode}):\n{stderr_data}\n", is_error=True)
                    self.app.master.after(0, self._update_progress_label, "依赖安装失败")
            except Exception as e:
                self.app.master.after(0, self._set_progress_determinate, 0)
                self.app._add_to_output_log_queue(f"安装依赖时出错: {e}\n", is_error=True)
                self.app.master.after(0, self._update_progress_label, "依赖安装出错")
            finally:
                self.app.master.after(2000, lambda: self._update_progress_label(original_progress_label_text))
                self.app.master.after(0, self._set_progress_determinate, 0) # Ensure it's reset

        threading.Thread(target=install, daemon=True).start()
        
    @protect_training_feature("batch_size")
    @protect_training_feature("epochs")
    def start_training(self):
        # (No changes to progress bar logic needed here; progress is for training process itself)
        if self.app.is_training: 
            messagebox.showinfo("训练中", "已有训练任务在进行中。", parent=self.app.master)
            return
        if not self.app.datasets_yaml_path.get() or self.app.datasets_yaml_path.get() == "未选择数据集文件":
            messagebox.showerror("错误", "请选择 datasets.yaml 文件。", parent=self.app.master)
            return
        if not os.path.exists(self.app.datasets_yaml_path.get()):
            messagebox.showerror("错误", f"datasets.yaml 文件未找到: {self.app.datasets_yaml_path.get()}", parent=self.app.master)
            return
        framework = self.app.selected_yolo_version_name.get()
        if not framework or framework not in config.YOLO_VERSIONS:
            messagebox.showerror("错误", "请选择有效的YOLO框架。", parent=self.app.master)
            return
        weights_path_str = self.app.weights_var.get()
        if not weights_path_str:
            messagebox.showerror("错误", "请指定预训练权重。", parent=self.app.master)
            return
        resolved_weights_path = weights_path_str
        if not os.path.isabs(weights_path_str):
            potential_path = os.path.join(self.weights_dir, weights_path_str)
            if os.path.exists(potential_path):
                resolved_weights_path = os.path.abspath(potential_path)
        if framework == "YOLOv5" and not os.path.exists(resolved_weights_path):
             messagebox.showerror("错误", f"YOLOv5权重文件未找到: {resolved_weights_path}\n(尝试从 '{self.weights_dir}' 解析或使用绝对路径)", parent=self.app.master)
             return
        device = self.app.device_var.get()
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available(): device = "0"
                else: device = "cpu"
                self.app._add_to_output_log_queue(f"自动选择设备: {device}\n")
            except ImportError: device = "cpu"; self.app._add_to_output_log_queue("自动选择设备: CPU (PyTorch未导入)\n")
            except Exception as e: device = "cpu"; self.app._add_to_output_log_queue(f"自动设备选择错误 ({e}), 使用CPU\n")
        try:
            epochs = int(self.app.epochs_var.get())
            batch_size = int(self.app.batch_var.get())
            imgsz = int(self.app.imgsz_var.get())
            workers = int(self.app.workers_var.get())
            if epochs <=0 or batch_size <=0 or imgsz <=0 or workers < 0:
                messagebox.showerror("参数错误", "Epochs, Batch Size, Image Size 必须为正数. Workers 必须为非负数.", parent=self.app.master)
                return
        except ValueError:
            messagebox.showerror("参数错误", "训练参数必须为有效数字。", parent=self.app.master)
            return
            
        # 直接检查训练参数是否超过许可证限制
        try:
            from security import check_feature_access, get_license_info
            # 检查轮次限制
            if not check_feature_access("epochs_limit", epochs):
                # 获取限制值
                try:
                    license_info = get_license_info()
                    max_epochs = license_info["features"]["epochs_limit"]
                    messagebox.showwarning("许可证限制", 
                                         f"免费版训练轮次限制为 {max_epochs}。将自动调整为{max_epochs}轮。\n升级到专业版可解除此限制。", 
                                         parent=self.app.master)
                    epochs = max_epochs
                    self.app.epochs_var.set(str(max_epochs))
                except Exception:
                    # 如果获取限制失败，使用默认值
                    epochs = 50
                    self.app.epochs_var.set("50")
            
            # 检查批量大小限制
            if not check_feature_access("batch_size_limit", batch_size):
                # 获取限制值
                try:
                    license_info = get_license_info()
                    max_batch = license_info["features"]["batch_size_limit"]
                    messagebox.showwarning("许可证限制", 
                                         f"免费版批量大小限制为 {max_batch}。将自动调整为{max_batch}。\n升级到专业版可解除此限制。", 
                                         parent=self.app.master)
                    batch_size = max_batch
                    self.app.batch_var.set(str(max_batch))
                except Exception:
                    # 如果获取限制失败，使用默认值
                    batch_size = 16
                    self.app.batch_var.set("16")
        except ImportError:
            # 如果安全模块不存在，继续执行
            pass
            
        project_dir_abs = os.path.abspath(self.app.project_dir_var.get())
        os.makedirs(project_dir_abs, exist_ok=True)
        run_name = self.app.run_name_var.get().strip() or "exp"
        yolo_config_dict = config.YOLO_VERSIONS[framework]
        command_generator = yolo_config_dict["command_generator"]
        code_path_val = self.app.yolo_code_path.get()
        train_script_abs_path = ""
        if framework == "YOLOv5":
            if not code_path_val or not os.path.isdir(code_path_val):
                messagebox.showerror("错误", "请先为YOLOv5设置有效的代码路径。", parent=self.app.master)
                return
            train_script_rel_path = yolo_config_dict.get("train_script_rel_path", "train.py")
            train_script_abs_path = os.path.join(code_path_val, train_script_rel_path)
            if not os.path.exists(train_script_abs_path):
                messagebox.showerror("错误", f"YOLOv5训练脚本未找到: {train_script_abs_path}", parent=self.app.master)
                return
        params = {
            "imgsz": imgsz, "batch_size": batch_size, "epochs": epochs,
            "datasets_yaml_path": self.app.datasets_yaml_path.get(),
            "weights_path": resolved_weights_path, 
            "device": device, "workers": workers,
            "project_dir_abs_path": project_dir_abs, "run_name": run_name,
            "yolo_code_path": code_path_val, 
            "train_script_abs_path": train_script_abs_path
        }
        try: training_command = command_generator(params)
        except Exception as e:
            messagebox.showerror("命令生成错误", f"生成训练命令时出错: {e}", parent=self.app.master)
            return
        working_dir = code_path_val if framework == "YOLOv5" and code_path_val else os.getcwd()
        self.app._add_to_output_log_queue("="*50 + "\n", is_train=True)
        self.app._add_to_output_log_queue(f"开始训练: {framework} {self.app.selected_yolo_subversion.get() or ''}\n", is_train=True)
        self.app._add_to_output_log_queue(f"命令: {' '.join(map(str,training_command))}\n", is_train=True)
        self.app._add_to_output_log_queue(f"工作目录: {working_dir}\n", is_train=True)
        self.app._add_to_output_log_queue("="*50 + "\n\n", is_train=True)
        self.app.is_training = True 
        self.stop_training_flag.clear()
        self.app.start_train_btn.config(state="disabled")
        self.app.stop_train_btn.config(state="normal")
        self.app.progress_bar["value"] = 0
        self.app.progress_label.config(text="训练初始化...")
        threading.Thread(target=self._run_training_thread, args=(training_command, working_dir), daemon=True).start()

    def _run_training_thread(self, command, working_dir):
        # (No changes needed here for progress bar logic)
        try:
            self.training_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True, cwd=working_dir,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
                encoding='utf-8', errors='replace'
            )
            for line in iter(self.training_process.stdout.readline, ''):
                if self.stop_training_flag.is_set():
                    self.app._add_to_output_log_queue("训练停止指令已接收...\n", is_train=True)
                    break
                self.app._add_to_output_log_queue(line, is_train=True)
                self.app.master.after(0, self._parse_training_progress, line)
            self.training_process.stdout.close()
            return_code = self.training_process.wait()
            if self.stop_training_flag.is_set(): self.app.master.after(0, self._training_finished, -1)
            else: self.app.master.after(0, self._training_finished, return_code)
        except FileNotFoundError as e:
            self.app.master.after(0, self._training_finished, -98, f"Executable or script not found: {e.filename}. Command: {' '.join(map(str,command))}")
        except Exception as e:
            self.app.master.after(0, self._training_finished, -99, str(e))
        finally: self.training_process = None

    def _parse_training_progress(self, line):
        # (No changes needed here)
        epoch_match = re.search(r'\b(\d+)/(\d+)\b', line)
        if epoch_match:
            try:
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                expected_total_epochs = int(self.app.epochs_var.get()) 
                if total_epochs > 0 and (total_epochs == expected_total_epochs or "Epoch" in line or "epoch" in line or current_epoch <= expected_total_epochs) :
                    progress = (current_epoch / total_epochs) * 100 if total_epochs > 0 else 0
                    self.app.progress_bar["value"] = progress
                    eta_match = re.search(r'ETA: (\S+)', line, re.IGNORECASE)
                    lr_match = re.search(r'lr: ([\d.eE+-]+)', line, re.IGNORECASE) 
                    loss_match = re.search(r'loss: ([\d.eE+-]+)', line, re.IGNORECASE) 
                    progress_text = f"Epoch: {current_epoch}/{total_epochs} ({progress:.1f}%)"
                    if eta_match: progress_text += f" ETA: {eta_match.group(1)}"
                    if lr_match: progress_text += f" LR: {lr_match.group(1)}"
                    if loss_match: progress_text += f" Loss: {loss_match.group(1)}"
                    self.app.progress_label.config(text=progress_text)
                    return 
            except ValueError: pass 
        scan_match = re.search(r'(Scanning [a-zA-Z\s()]+|Caching RAM|Scanning dataset)\s*\((\d+)%\)', line, re.IGNORECASE)
        if scan_match:
            try:
                action = scan_match.group(1).strip()
                percent = int(scan_match.group(2))
                self.app.progress_bar["value"] = percent
                self.app.progress_label.config(text=f"{action}: {percent}%")
                return
            except ValueError: pass

    def _training_finished(self, return_code, error_msg=None):
        # (No changes needed here)
        self.app.is_training = False 
        self.app.start_train_btn.config(state="normal")
        self.app.stop_train_btn.config(state="disabled")
        if error_msg:
            self.app._add_to_output_log_queue(f"\n训练异常: {error_msg}\n", is_error=True, is_train=True)
            self.app.progress_label.config(text="训练异常!")
            self.app.progress_bar["value"] = 0
        elif return_code == 0:
            self.app._add_to_output_log_queue("\n训练成功完成。\n", is_train=True)
            self.app.progress_label.config(text="训练成功完成!")
            self.app.progress_bar["value"] = 100
        elif return_code == -1: 
            self.app._add_to_output_log_queue("\n训练已由用户停止。\n", is_train=True)
            self.app.progress_label.config(text="训练已停止")
        elif return_code == -98: 
             self.app._add_to_output_log_queue(f"\n训练启动失败: {error_msg}\n", is_error=True, is_train=True) 
             self.app.progress_label.config(text="训练启动失败 (文件未找到)")
             self.app.progress_bar["value"] = 0
        else: 
            self.app._add_to_output_log_queue(f"\n训练失败，错误码: {return_code}\n", is_error=True, is_train=True)
            self.app.progress_label.config(text=f"训练失败 (错误码: {return_code})")
            self.app.progress_bar["value"] = 0

    def stop_training(self):
        # (No changes needed here)
        if self.app.is_training and self.training_process: 
            self.app._add_to_output_log_queue("尝试停止训练...\n")
            self.stop_training_flag.set()
            try:
                self.training_process.terminate() 
            except ProcessLookupError: self.app._add_to_output_log_queue("进程已结束。\n")
            except Exception as e:
                self.app._add_to_output_log_queue(f"停止训练时发送terminate信号出错: {e}\n", is_error=True)
                try:
                    self.training_process.kill()
                    self.app._add_to_output_log_queue("进程已被强制终止 (kill)。\n")
                except Exception as e_kill:
                    self.app._add_to_output_log_queue(f"强制终止进程时出错: {e_kill}\n", is_error=True)
        else: self.app._add_to_output_log_queue("没有正在进行的训练任务可停止。\n")
        self.app.stop_train_btn.config(state="disabled")
        self.save_checkpoint()

    def save_checkpoint(self):
        """保存训练断点（权重、参数、日志等）"""
        # 这里只保存权重和参数路径，实际可根据YOLO实现扩展
        try:
            # 假设YOLO训练脚本会自动保存最新权重到 runs/train/exp*/last.pt
            # 这里只做路径记录
            project_dir = os.path.abspath(self.app.project_dir_var.get())
            run_name = self.app.run_name_var.get().strip() or "exp"
            exp_dir = os.path.join(project_dir, run_name)
            last_pt = os.path.join(exp_dir, "weights", "last.pt")
            if os.path.exists(last_pt):
                # 记录断点信息
                checkpoint_info = {
                    "weights": last_pt,
                    "params": {
                        "imgsz": self.app.imgsz_var.get(),
                        "batch_size": self.app.batch_var.get(),
                        "epochs": self.app.epochs_var.get(),
                        "datasets_yaml_path": self.app.datasets_yaml_path.get(),
                        "device": self.app.device_var.get(),
                        "workers": self.app.workers_var.get(),
                        "project_dir": project_dir,
                        "run_name": run_name,
                    }
                }
                import json
                ckpt_path = os.path.join(self.checkpoint_dir, f"{run_name}_checkpoint.json")
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_info, f, ensure_ascii=False, indent=2)
                self.last_checkpoint_path = ckpt_path
                self.app._add_to_output_log_queue(f"已保存训练断点: {ckpt_path}", is_train=True)
            else:
                self.app._add_to_output_log_queue("未找到最新权重，断点未保存。", is_error=True, is_train=True)
        except Exception as e:
            self.app._add_to_output_log_queue(f"保存断点失败: {e}", is_error=True, is_train=True)

    def resume_training(self):
        """从最近断点恢复训练，仅专业版可用"""
        try:
            from security import is_pro_version
            if not is_pro_version():
                self.app._add_to_output_log_queue("断点恢复训练为专业版专属功能，请激活专业版。", is_error=True, is_train=True)
                return
        except ImportError:
            self.app._add_to_output_log_queue("安全模块未加载，无法校验专业版。", is_error=True, is_train=True)
            return
        import glob, json
        # 查找最近的断点文件
        ckpt_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, "*_checkpoint.json")), key=os.path.getmtime, reverse=True)
        if not ckpt_files:
            self.app._add_to_output_log_queue("未找到可用的训练断点。", is_error=True, is_train=True)
            return
        ckpt_path = ckpt_files[0]
        with open(ckpt_path, "r", encoding="utf-8") as f:
            checkpoint_info = json.load(f)
        # 恢复参数
        params = checkpoint_info["params"]
        self.app.imgsz_var.set(params["imgsz"])
        self.app.batch_var.set(params["batch_size"])
        self.app.epochs_var.set(params["epochs"])
        self.app.datasets_yaml_path.set(params["datasets_yaml_path"])
        self.app.device_var.set(params["device"])
        self.app.workers_var.set(params["workers"])
        self.app.project_dir_var.set(params["project_dir"])
        self.app.run_name_var.set(params["run_name"])
        # 设置权重为断点权重
        self.app.weights_var.set(checkpoint_info["weights"])
        self.app._add_to_output_log_queue(f"已加载断点参数，准备恢复训练...", is_train=True)
        self.start_training()