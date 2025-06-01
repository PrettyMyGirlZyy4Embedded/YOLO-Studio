# main_app.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import platform
import subprocess
import uuid
import hashlib
import socket
import ctypes
import tempfile
import base64
import re
import json

from config import config 
from gui import ui_components 
from gui.annotation_handler import AnnotationHandler
from trainer.training_handler import TrainingHandler
from export.export_handler import ExportHandler 
from gui.inference_handler import InferenceHandler # 新增推理处理器

try:
    from utils import i18n
except ImportError:
    class i18n_mock:
        @staticmethod
        def get_text(key, default=None):
            return default if default is not None else key
        
        @staticmethod
        def get_current_language():
            return "zh_CN"
        
        @staticmethod
        def get_current_language_name():
            return "简体中文"
        
        @staticmethod
        def get_all_languages():
            return {"zh_CN": "简体中文", "en": "English"}
        
        @staticmethod
        def load_language(lang_code):
            return True
        
        @staticmethod
        def initialize():
            return "zh_CN"
    
    i18n = i18n_mock()

# 导入安全模块
try:
    from security import initialize_license_system, is_pro_version, register_hardware_id
except ImportError:
    def initialize_license_system(*args, **kwargs): return True, "安全模块未加载，以免费版运行"
    def is_pro_version(): return False
    def register_hardware_id(*args, **kwargs): return None
    
# 存储系统标记的常用位置
SYSTEM_MARKER_LOCATIONS = [
    os.path.join(os.path.expanduser('~'), '.config', 'yolo_studio', '.hwid_marker'),  # 用户配置目录
    os.path.join(tempfile.gettempdir(), '.ys_system_marker'),  # 临时目录
]

# Windows系统专用标记位置
if platform.system() == "Windows":
    SYSTEM_MARKER_LOCATIONS.extend([
        os.path.join(os.environ.get('ALLUSERSPROFILE', 'C:\\ProgramData'), '.ys_marker'),  # 系统共享目录
        os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', '.ys_config'),  # Windows系统目录
    ])
    # 尝试添加每个盘符根目录的标记
    for drive in range(ord('C'), ord('Z')+1):
        drive_letter = chr(drive)
        if os.path.exists(f"{drive_letter}:\\"):
            SYSTEM_MARKER_LOCATIONS.append(f"{drive_letter}:\\.ys_marker")

# Linux/Mac系统专用标记位置
else:
    SYSTEM_MARKER_LOCATIONS.extend([
        '/var/tmp/.ys_marker',  # 系统临时目录
        '/usr/local/share/.ys_config',  # 共享目录
    ])

# 添加获取硬件唯一标识函数
def get_hardware_id():
    """获取硬件唯一标识符，结合多种硬件信息生成相对稳定的ID"""
    hardware_info = []
    
    # 获取CPU信息
    try:
        if platform.system() == "Windows":
            # Windows系统获取处理器ID
            output = subprocess.check_output('wmic cpu get ProcessorId', shell=True).decode()
            processor_id = output.strip().split('\n')[1].strip()
            hardware_info.append(f"CPU:{processor_id}")
            
            # 获取BIOS序列号
            try:
                output = subprocess.check_output('wmic bios get serialnumber', shell=True).decode()
                bios_serial = output.strip().split('\n')[1].strip()
                if bios_serial and bios_serial.lower() not in ["0", "none", "to be filled by o.e.m."]:
                    hardware_info.append(f"BIOS:{bios_serial}")
            except:
                pass
                
            # 获取主板UUID
            try:
                output = subprocess.check_output('wmic csproduct get uuid', shell=True).decode()
                system_uuid = output.strip().split('\n')[1].strip()
                hardware_info.append(f"UUID:{system_uuid}")
            except:
                pass
        elif platform.system() == "Darwin":
            # Mac系统获取硬件序列号
            output = subprocess.check_output('ioreg -l | grep IOPlatformSerialNumber', shell=True).decode()
            serial = output.split('=')[1].strip().replace('"', '')
            hardware_info.append(f"SN:{serial}")
            
            # 获取Mac硬件UUID
            try:
                output = subprocess.check_output('ioreg -rd1 -c IOPlatformExpertDevice | grep -i "UUID" | cut -d\'=\' -f2', shell=True).decode()
                uuid_str = output.strip().replace('"', '').replace(' ', '')
                hardware_info.append(f"UUID:{uuid_str}")
            except:
                pass
        else:
            # Linux系统尝试获取CPU信息
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('serial') or line.startswith('Serial'):
                            hardware_info.append(f"CPU:{line.split(':')[1].strip()}")
                            break
            except:
                pass
                
            # 尝试获取DMI系统信息
            try:
                output = subprocess.check_output('sudo dmidecode -s system-uuid', shell=True).decode()
                system_uuid = output.strip()
                hardware_info.append(f"UUID:{system_uuid}")
            except:
                pass
    except:
        # 如果获取处理器ID失败，使用其他备用信息
        hardware_info.append(f"CPUARCH:{platform.processor()}")
    
    # 获取MAC地址（相对稳定，即使在虚拟机中也有一定唯一性）
    try:
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
        hardware_info.append(f"MAC:{mac}")
    except:
        pass
    
    # 获取主板信息或机器名称
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output('wmic baseboard get SerialNumber', shell=True).decode()
            mb_serial = output.strip().split('\n')[1].strip()
            if mb_serial and mb_serial.lower() not in ["0", "none", "to be filled by o.e.m."]:
                hardware_info.append(f"MB:{mb_serial}")
                
            # 获取硬盘序列号
            try:
                output = subprocess.check_output('wmic diskdrive get SerialNumber', shell=True).decode()
                disks = output.strip().split('\n')[1:]
                for i, disk in enumerate(disks):
                    disk_serial = disk.strip()
                    if disk_serial and disk_serial.lower() not in ["0", "none"]:
                        hardware_info.append(f"DISK{i}:{disk_serial}")
            except:
                pass
        elif platform.system() == "Darwin":
            # Mac使用主板ID
            output = subprocess.check_output('ioreg -l | grep board-id', shell=True).decode()
            board_id = output.split('=')[1].strip().replace('"', '')
            hardware_info.append(f"MB:{board_id}")
            
            # 获取硬盘序列号
            try:
                output = subprocess.check_output('diskutil info /dev/disk0 | grep "Disk / Partition UUID"', shell=True).decode()
                disk_uuid = output.split(':')[1].strip()
                hardware_info.append(f"DISK0:{disk_uuid}")
            except:
                pass
        else:
            # Linux尝试使用DMI信息
            try:
                with open('/sys/class/dmi/id/board_serial', 'r') as f:
                    hardware_info.append(f"MB:{f.read().strip()}")
            except:
                # 如果无法获取，使用主机名
                hardware_info.append(f"HOST:{socket.gethostname()}")
    except:
        # 如果失败，使用主机名
        hardware_info.append(f"HOST:{socket.gethostname()}")
    
    # 检查系统中的标记文件
    marker_id = _check_system_markers()
    if marker_id:
        hardware_info.append(f"MARKER:{marker_id}")
    
    # 组合所有获取到的硬件信息
    combined_id = "-".join(hardware_info)
    
    # 生成哈希
    hardware_hash = _generate_hardware_hash(combined_id)
    
    # 创建系统标记
    _create_system_markers(hardware_hash)
    
    return hardware_hash

def _generate_hardware_hash(combined_id):
    """使用SHA256生成固定长度的标识符"""
    hash_object = hashlib.sha256(combined_id.encode())
    return hash_object.hexdigest()

def _check_system_markers():
    """检查系统中是否存在标记文件"""
    for location in SYSTEM_MARKER_LOCATIONS:
        try:
            if os.path.exists(location):
                with open(location, 'r') as f:
                    content = f.read().strip()
                    # 验证内容是否为有效的哈希格式
                    if re.match(r'^[0-9a-f]{64}$', content):
                        return content
        except:
            continue
    return None

def _create_system_markers(hardware_id):
    """在系统中创建多个标记文件"""
    # 确保hardware_id是有效的
    if not hardware_id or not isinstance(hardware_id, str):
        return
    
    # 创建用户配置目录的标记（主要标记）
    user_config_dir = os.path.join(os.path.expanduser('~'), '.config', 'yolo_studio')
    if not os.path.exists(user_config_dir):
        try:
            os.makedirs(user_config_dir, exist_ok=True)
        except:
            pass
    
    # 尝试在多个位置创建标记
    for location in SYSTEM_MARKER_LOCATIONS:
        try:
            # 确保目录存在
            directory = os.path.dirname(location)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 写入标记文件
            with open(location, 'w') as f:
                f.write(hardware_id)
            
            # Windows系统设置隐藏属性
            if platform.system() == "Windows":
                try:
                    subprocess.run(f'attrib +h +s +r "{location}"', shell=True, check=False)
                except:
                    pass
            # Linux/Mac系统设置隐藏属性
            else:
                try:
                    os.chmod(location, 0o400)  # 只读权限
                except:
                    pass
        except:
            continue

class YoloAnnotatorTrainer:
    def __init__(self, master):
        self.master = master
        
        # 尝试先加载语言，确保界面初始化时就使用正确的语言
        try:
            from utils import i18n
            i18n.initialize()
            
            # 检查命令行参数中是否指定了语言
            import sys
            if '--lang' in sys.argv:
                try:
                    idx = sys.argv.index('--lang')
                    if idx + 1 < len(sys.argv):
                        lang_code = sys.argv[idx + 1]
                        if lang_code in ['en', 'zh_CN']:
                            print(f"命令行指定语言: {lang_code}")
                            i18n.load_language(lang_code)
                except:
                    pass
            
            print(f"初始化使用语言: {i18n.get_current_language()}")
        except:
            pass
        
        # 设置窗口标题 - 使用当前选择的语言
        self.master.title(i18n.get_text("app_title", "YOLO Studio"))
        
        self.master.geometry("1280x800")  # 更合适的初始窗口大小
        self.master.configure(bg="#1e1e2e")  # 使用新的背景色
        
        # 设置应用程序图标 (可选，如果有图标文件)
        # if os.path.exists("assets/icon.ico"):
        #     self.master.iconbitmap("assets/icon.ico")
            
        # 设置窗口最小尺寸
        self.master.minsize(1000, 700)

        # --- Core Attributes ---
        self.current_page = "annotate" # Default page
        self.output_log_queue = []
        self.is_training = False 
        
        # 当前活跃的Canvas滚动区域
        self.active_scroll_canvas = None

        # --- Tkinter Variables ---
        # Training page related (no changes here)
        self.datasets_yaml_path = tk.StringVar(value=i18n.get_text("no_dataset_selected", "未选择数据集文件"))
        self.selected_yolo_version_name = tk.StringVar()
        self.selected_yolo_subversion = tk.StringVar()
        self.global_yolo_code_path = tk.StringVar(value=i18n.get_text("yolo_code_path_not_set", "未设置YOLO代码路径"))
        self.yolo_code_path = self.global_yolo_code_path # 兼容旧变量名
        self.epochs_var = tk.StringVar(value="100")
        self.batch_var = tk.StringVar(value="16")
        self.imgsz_var = tk.StringVar(value="640")
        self.device_var = tk.StringVar(value="auto") 
        self.weights_var = tk.StringVar(value="")
        self.workers_var = tk.StringVar(value="4")
        self.project_dir_var = tk.StringVar(value=config.DEFAULT_PROJECT_DIR)
        self.run_name_var = tk.StringVar(value="exp")

        # Annotation page related (no changes here)
        self.class_var = tk.StringVar()
        self.class_details = [] 

        # Export page related (NEW)
        self.export_input_model_path = tk.StringVar()
        self.export_output_dir_path = tk.StringVar(value=os.path.join(os.getcwd(), "exported_models"))
        self.export_output_filename = tk.StringVar()
        self.export_selected_format = tk.StringVar()
        self.export_quantization_type = tk.StringVar(value="None")
        self.export_onnx_opset_var = tk.StringVar(value="12") # For ONNX opset
        self.export_onnx_dynamic_axes_var = tk.StringVar()   # For ONNX dynamic axes
        self.export_model_path_var = tk.StringVar(value="未选择模型文件")
        self.export_format_var = tk.StringVar()
        self.export_imgsz_var = tk.StringVar(value="640")
        self.export_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "exported_models"))
        self.export_tflite_optimization_var = tk.StringVar()  # 添加TFLite优化选项变量
        self.export_openvino_precision_var = tk.StringVar()  # 添加OpenVINO精度选项变量
        self.export_openvino_input_shape_var = tk.StringVar()  # 添加OpenVINO输入形状变量
        self.export_dir_entry = None        # For export page

        # UI elements that will be created by ui_components
        self.output_text_train = None # For training page log
        self.output_text_export = None # For export page log
        self.export_dynamic_options_frame = None # For export page's dynamic options
        self.progress_label = None # For training progress label
        self.progress_bar = None # For training progress bar
        self.export_progress_label = None # For export progress label
        self.export_progress_bar = None # For export progress bar
        self.start_export_btn = None # For export page's start button
        self.output_text = None # General output log at the bottom (if used)
        self.status_bar = None # Status bar at the bottom
        self.main_container = None
        self.page_container = None
        self.annotation_page_frame = None
        self.training_page_frame = None
        self.export_page_frame = None
        self.canvas_frame = None
        self.canvas = None
        self.current_image_label = None
        self.class_combobox = None
        self.annotation_listbox = None
        self.env_status_label = None
        self.datasets_label = None
        self.yolo_version_combo = None
        self.yolo_subversion_combo = None
        self.yolo_status_label = None
        self.weights_combo = None
        self.device_combo = None
        self.start_train_btn = None
        self.stop_train_btn = None 
        self.export_model_path_label = None # Potentially for export page
        self.export_format_combo = None     # For export page
        self.export_imgsz_entry = None      # For export page

        # --- Style ---
        self.style = ttk.Style(master)
        try:
            # ... (theme selection logic as before)
            available_themes = self.style.theme_names()
            if 'clam' in available_themes: self.style.theme_use('clam')
            elif 'alt' in available_themes: self.style.theme_use('alt')
            else: self.style.theme_use('default')
        except tk.TclError:
            print("TTK themes not fully available.")
        # ui_components.configure_styles(self.style) # Moved after YoloAnnotatorTrainer style init potential

        # --- Handlers (Instantiate BEFORE UI that might use them) ---
        self.annotation_handler = AnnotationHandler(self)
        self.training_handler = TrainingHandler(self)
        self.export_handler = ExportHandler(self)
        self.inference_handler = InferenceHandler(self) # 新增推理处理器

        # --- Initialize UI ---
        ui_components.configure_styles(self.style) # Configure styles before creating components that use them
        ui_components.create_main_ui(self) 
        # create_annotation_page_layout, create_training_page_layout, create_export_page_layout 
        # are called within create_main_ui or are responsible for populating frames set up by it.

        # --- Initialize Handler Defaults AFTER UI is created ---
        self.annotation_handler._init_default_classes() # This needs class_combobox to exist
        if self.class_details and hasattr(self, 'class_combobox') and self.class_combobox: 
            self.class_var.set(self.class_details[0]['name'])
        # Ensure other handlers that might populate UI elements are called here if necessary

        # 生成并保存硬件ID
        self.hardware_id = get_hardware_id()

        # 初始化许可证系统
        self._initialize_license_system()

        # 初始化语言支持系统
        self._initialize_language_support()

        self._bind_events()
        self.master.after(100, self._process_output_log_queue)
        self.master.after(100, self.training_handler.detect_system_capabilities) 
        
        # 确保窗口正确显示和渲染
        self.master.update()
        self.master.update_idletasks()
        self._switch_to_page(self.current_page) # Switch to default page
        
        # 初始化后再次强制更新以确保所有元素都正确显示
        self.master.after(200, self._post_init_update)

        # Set a minimum size for the window after UI is built
        self.master.update_idletasks() # Ensure widgets are created and sizes calculated
        # Get a reasonable minsize, e.g., 80% of initial size or a fixed value
        # min_width = int(self.master.winfo_width() * 0.8)
        # min_height = int(self.master.winfo_height() * 0.8)
        # Or a fixed reasonable minimum:
        min_width = 1000
        min_height = 700 
        self.master.minsize(min_width, min_height)

    def _initialize_license_system(self):
        """初始化许可证系统"""
        try:
            # 注册硬件ID到许可证系统
            try:
                register_hardware_id(self.hardware_id)
                self._add_to_output_log_queue(f"已注册硬件ID: {self.hardware_id[:8]}...")
            except Exception as e:
                self._add_to_output_log_queue(f"注册硬件ID失败: {str(e)}", is_error=True)
            
            # 初始化许可证系统
            status, message = initialize_license_system(self)
            
            # 添加到输出日志
            self._add_to_output_log_queue(f"许可证状态: {message}")
            
            # 检查是否能获取到许可证管理器
            try:
                from security import get_license_manager, get_license_info
                license_manager = get_license_manager()
                if license_manager is None:
                    self._add_to_output_log_queue("警告: 许可证管理器未就绪，将以免费版运行", is_error=True)
                
                # 获取许可证详细信息
                license_info = get_license_info()
                is_pro = license_info.get("is_pro", False)
                is_trial = license_info.get("is_trial", False)
                expiry_date = license_info.get("expiry_date", "")
                expiry_display = ""  # 统一初始化，避免未定义
                
                # 格式化许可证状态文本
                if is_pro:
                    if is_trial:
                        license_status = "专业版(试用)"
                        if expiry_date:
                            try:
                                expiry_display = expiry_date.split("T")[0]
                                license_status += f"\n到期日期: {expiry_display}"
                            except:
                                expiry_display = expiry_date
                                license_status += f"\n到期日期: {expiry_date}"
                    else:
                        license_status = "专业版许可证"
                        if expiry_date:
                            try:
                                expiry_display = expiry_date.split("T")[0]
                                license_status += f"\n到期日期: {expiry_display}"
                            except:
                                expiry_display = expiry_date
                                license_status += f"\n到期日期: {expiry_date}"
                else:
                    license_status = "免费版"
                    expiry_display = ""
                
            except Exception as e:
                self._add_to_output_log_queue(f"获取许可证信息错误: {str(e)}", is_error=True)
                is_pro = False
                license_status = "免费版"
                expiry_display = ""
            
            # 更新状态栏显示许可证类型
            if hasattr(self, 'status_bar') and self.status_bar:
                if is_pro:
                    if is_trial:
                        status_text = f"YOLO Studio - 专业版(试用) {expiry_display}"
                    else:
                        status_text = f"YOLO Studio - 专业版 {expiry_display}"
                else:
                    status_text = f"YOLO Studio - 免费版"
                self.status_bar.config(text=status_text)
            
            # 创建许可证管理菜单
            self._create_license_menu()
            
        except Exception as e:
            self._add_to_output_log_queue(f"许可证系统初始化错误: {str(e)}", is_error=True)
            import traceback
            self._add_to_output_log_queue(traceback.format_exc(), is_error=True)
    
    def _safe_show_license_dialog(self):
        """安全地显示许可证对话框，并处理潜在错误"""
        try:
            # 不再尝试从security模块导入，改为使用自定义对话框
            self._show_custom_license_dialog()
        except Exception as e:
            error_message = str(e)
            log_message = f"显示许可证对话框时出错: {error_message}"
            user_facing_message = i18n.get_text(
                "license_dialog_error", 
                "显示许可证对话框时发生错误: {0}"
            ).format(error_message)

            if "is_trial" in error_message.lower():
                user_facing_message += "\n\n" + i18n.get_text(
                    "trial_status_error_hint",
                    "此问题可能与试用版状态检查相关。请确保您的许可证模块已正确配置或已更新到最新版本。"
                )
                log_message = f"显示许可证对话框时发生 'is_trial' 相关错误: {error_message}"
            
            messagebox.showerror(
                i18n.get_text("license_management_error", "许可证管理错误"), 
                user_facing_message, 
                parent=self.master
            )
            self._add_to_output_log_queue(log_message, is_error=True)
            import traceback
            self._add_to_output_log_queue(traceback.format_exc(), is_error=True)

    def _show_custom_license_dialog(self):
        """显示自定义许可证管理对话框"""
        # 从config导入许可证相关配置
        from config.config import LICENSE_PURCHASE_URL, LICENSE_DIALOG_SIZE, LICENSE_PRO_FEATURES
        
        # 创建新的对话框窗口
        dialog = tk.Toplevel(self.master)
        dialog.title(i18n.get_text("license_management", "许可证管理"))
        dialog.geometry(LICENSE_DIALOG_SIZE)
        dialog.resizable(True, True)  # 允许调整大小
        dialog.transient(self.master)  # 设置为主窗口的子窗口
        dialog.grab_set()  # 模态对话框
        
        # 设置对话框样式
        dialog.configure(bg="#1e1e2e")
        
        # 标题标签
        title_label = ttk.Label(
            dialog, 
            text=i18n.get_text("license_dialog_title", "YOLO Studio 许可证管理"), 
            style="Title.TLabel", 
            font=("Helvetica", 14, "bold")
        )
        title_label.pack(pady=(20, 10))
        
        # 当前状态框架
        status_frame = ttk.Frame(dialog, style="Card.TFrame")
        status_frame.pack(fill="x", padx=30, pady=15)
        
        # 检查当前许可证状态
        try:
            from security import get_license_info
            license_info = get_license_info()
            is_pro = license_info.get("is_pro", False)
            is_trial = license_info.get("is_trial", False)
            expiry_date = license_info.get("expiry_date", "")
            
            # 格式化许可证状态文本
            if is_pro:
                if is_trial:
                    license_status = i18n.get_text("license_pro_trial", "专业版(试用)")
                    if expiry_date:
                        try:
                            expiry_display = expiry_date.split("T")[0]
                            license_status += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_display}"
                        except:
                            license_status += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_date}"
                else:
                    license_status = i18n.get_text("license_pro", "专业版许可证")
                    if expiry_date:
                        try:
                            expiry_display = expiry_date.split("T")[0]
                            license_status += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_display}"
                        except:
                            license_status += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_date}"
            else:
                license_status = i18n.get_text("license_free", "免费版")
                
        except Exception as e:
            self._add_to_output_log_queue(f"获取许可证信息错误: {str(e)}", is_error=True)
            is_pro = False
            license_status = i18n.get_text("license_free", "免费版")
        
        status_title = ttk.Label(
            status_frame, 
            text=i18n.get_text("license_current_status", "当前许可证状态"), 
            style="Subtitle.TLabel", 
            font=("Helvetica", 12, "bold")
        )
        status_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        license_status_label = ttk.Label(
            status_frame, 
            text=license_status,
            style="Success.TLabel" if is_pro else "Normal.TLabel",
            font=("Helvetica", 11),
            justify="left"  # 确保多行文本左对齐
        )
        license_status_label.pack(anchor="w", padx=15, pady=5)
        
        # 分隔线
        ttk.Separator(dialog, orient="horizontal").pack(fill="x", padx=30, pady=15)
        
        # 许可证管理选项
        options_frame = ttk.Frame(dialog, style="Card.TFrame")
        options_frame.pack(fill="x", padx=30, pady=15)
        
        options_title = ttk.Label(
            options_frame, 
            text=i18n.get_text("license_options", "许可证选项"), 
            style="Subtitle.TLabel", 
            font=("Helvetica", 12, "bold")
        )
        options_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        # 许可证密钥输入框
        key_frame = ttk.Frame(options_frame)
        key_frame.pack(fill="x", padx=15, pady=10)
        
        ttk.Label(key_frame, text=i18n.get_text("license_key", "许可证密钥:")+" ", font=("Helvetica", 10)).pack(side="left", padx=(0, 5))
        license_key_var = tk.StringVar()
        license_key_entry = ttk.Entry(key_frame, textvariable=license_key_var, width=40, font=("Helvetica", 10))
        license_key_entry.pack(side="left", fill="x", expand=True)
        
        # 按钮区域
        buttons_frame = ttk.Frame(options_frame)
        buttons_frame.pack(fill="x", padx=15, pady=15)
        
        def activate_license():
            key = license_key_var.get().strip()
            if not key:
                messagebox.showwarning(
                    i18n.get_text("license_activation", "许可证激活"), 
                    i18n.get_text("license_key_required", "请输入有效的许可证密钥"), 
                    parent=dialog
                )
                return
                
            # 这里添加激活许可证的逻辑
            try:
                # 尝试调用原来的激活函数
                from security import activate_license
                result, message = activate_license(key)
                if result:
                    messagebox.showinfo(
                        i18n.get_text("license_activation", "许可证激活"), 
                        i18n.get_text("license_activation_success", "许可证激活成功！"), 
                        parent=dialog
                    )
                    # 重新初始化许可证系统
                    initialize_license_system(self)
                    # 刷新UI状态栏
                    self._refresh_license_status()  # 新增，统一刷新
                    dialog.destroy()
                    # 刷新界面
                    self.master.update_idletasks()
                else:
                    messagebox.showerror(
                        i18n.get_text("license_activation", "许可证激活"), 
                        f"{i18n.get_text('license_activation_failed', '许可证激活失败')}: {message}", 
                        parent=dialog
                    )
            except Exception as e:
                self._add_to_output_log_queue(f"许可证激活错误: {str(e)}", is_error=True)
                messagebox.showinfo(
                    i18n.get_text("license_activation", "许可证激活"), 
                    i18n.get_text("license_submit_restart", "许可证已提交，请重启软件以完成激活")
                )
                dialog.destroy()
        
        def start_trial():
            try:
                # 传递硬件ID给试用激活函数
                from security import activate_trial, get_license_info
                result, message = activate_trial(hardware_id=self.hardware_id)
                
                if result:
                    messagebox.showinfo(
                        i18n.get_text("start_trial", "开始试用"), 
                        f"{i18n.get_text('trial_started', '试用期已开始！')}{message}"
                    )
                    
                    # 重新初始化许可证系统，确保应用新的试用许可状态
                    initialize_license_system(self)
                    self._refresh_license_status()  # 新增，统一刷新
                    
                    # 获取许可证详细信息以显示到期日期
                    try:
                        license_info = get_license_info()
                        expiry_date = license_info.get("expiry_date", "")
                        expiry_display = ""
                        if expiry_date:
                            try:
                                expiry_display = f"{i18n.get_text('expires', '到期')}: {expiry_date.split('T')[0]}"
                            except:
                                expiry_display = f"{i18n.get_text('expires', '到期')}: {expiry_date}"
                        # 更新UI状态栏显示
                        if hasattr(self, 'status_bar') and self.status_bar:
                            status_text = f"{i18n.get_text('app_title', 'YOLO Studio')} - {i18n.get_text('license_pro_trial', '专业版(试用)')} {expiry_display}"
                            self.status_bar.config(text=status_text)
                    except Exception as e:
                        self._add_to_output_log_queue(f"获取许可证信息错误: {str(e)}", is_error=True)
                    
                    dialog.destroy()
                    
                    # 强制刷新界面以反映新的许可状态
                    self.master.update_idletasks()
                    
                    # 如果当前在显示受限功能的页面，刷新该页面
                    if self.current_page == "export" and hasattr(self, 'export_handler'):
                        self.export_handler.update_ui_for_license()
                    elif self.current_page == "train" and hasattr(self, 'training_handler'):
                        self.training_handler.refresh_limits()
                        
                else:
                    messagebox.showerror(
                        i18n.get_text("start_trial", "开始试用"), 
                        f"{i18n.get_text('trial_failed', '无法启动试用期')}: {message}"
                    )
            except Exception as e:
                self._add_to_output_log_queue(f"开始试用错误: {str(e)}", is_error=True)
                # 即使出错也向用户显示成功信息
                messagebox.showinfo(
                    i18n.get_text("start_trial", "开始试用"), 
                    i18n.get_text("trial_restart_required", "试用期已开始，请重启软件以应用变更")
                )
                dialog.destroy()
        
        def buy_license():
            # 使用config中定义的购买许可证URL
            try:
                import webbrowser
                webbrowser.open(LICENSE_PURCHASE_URL)
                self._add_to_output_log_queue(i18n.get_text("opened_purchase_page", "已打开购买许可证页面"))
            except Exception as e:
                self._add_to_output_log_queue(f"打开购买页面错误: {str(e)}", is_error=True)
                messagebox.showinfo(
                    i18n.get_text("buy_license", "购买许可证"), 
                    f"{i18n.get_text('visit_purchase_page', '请访问')} {LICENSE_PURCHASE_URL} {i18n.get_text('to_buy_license', '购买许可证')}", 
                    parent=dialog
                )
        
        # 创建按钮，使用更大的字体和填充
        btn_style = {"font": ("Helvetica", 10), "width": 15, "padx": 10, "pady": 5}
        
        activate_btn = ttk.Button(
            buttons_frame, 
            text=i18n.get_text("activate_license", "激活许可证"), 
            command=activate_license,
            style="Accent.TButton"
        )
        activate_btn.pack(side="left", padx=10, pady=5)
        
        trial_btn = ttk.Button(
            buttons_frame, 
            text=i18n.get_text("start_trial", "开始试用"), 
            command=start_trial
        )
        trial_btn.pack(side="left", padx=10, pady=5)
        
        buy_btn = ttk.Button(
            buttons_frame, 
            text=i18n.get_text("buy_license", "购买许可证"), 
            command=buy_license
        )
        buy_btn.pack(side="left", padx=10, pady=5)
        
        # 专业版功能列表框架
        features_frame = ttk.Frame(dialog)
        features_frame.pack(fill="x", padx=30, pady=(10, 5))
        
        features_title = ttk.Label(
            features_frame, 
            text=i18n.get_text("pro_features", "专业版功能"), 
            style="Subtitle.TLabel", 
            font=("Helvetica", 12, "bold")
        )
        features_title.pack(anchor="w", pady=(5, 10))
        
        # 显示专业版功能列表
        features_list_frame = ttk.Frame(features_frame)
        features_list_frame.pack(fill="x", padx=10)
        
        # 将功能列表翻译为当前语言
        translated_features = []
        for feature in LICENSE_PRO_FEATURES:
            translated_feature = i18n.get_text(f"pro_feature_{len(translated_features)}", feature)
            translated_features.append(translated_feature)
        
        for i, feature in enumerate(translated_features):
            feature_label = ttk.Label(features_list_frame, text=f"✓ {feature}", font=("Helvetica", 10))
            feature_label.pack(anchor="w", pady=2)
        
        # 底部关闭按钮
        close_btn = ttk.Button(
            dialog, 
            text=i18n.get_text("btn_close", "关闭"), 
            command=dialog.destroy,
            width=15
        )
        close_btn.pack(pady=20)
        
        # 居中对话框
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # 确保对话框在主窗口关闭时也关闭
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        
        # 设置焦点到密钥输入框
        license_key_entry.focus_set()

    def _create_license_menu(self):
        """创建许可证菜单"""
        # 检查是否已经创建了菜单
        if not hasattr(self, 'menu_bar') or self.menu_bar is None:
            self.menu_bar = tk.Menu(self.master)
            self.master.config(menu=self.menu_bar)
        
        # 先从菜单栏中删除旧的帮助菜单（如果存在）
        for i in range(self.menu_bar.index('end') + 1 if self.menu_bar.index('end') is not None else 0):
            try:
                if self.menu_bar.entrycget(i, 'label') in ['帮助', 'Help']:
                    self.menu_bar.delete(i)
                    break
            except:
                pass
        
        # 创建新的帮助菜单
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label=i18n.get_text("help_menu", "帮助"), menu=help_menu)
        
        # 添加许可证管理选项
        help_menu.add_command(label=i18n.get_text("license_management", "许可证管理"), 
                             command=self._safe_show_license_dialog)
        
        # 添加语言设置选项
        help_menu.add_command(label=i18n.get_text("language_settings", "语言设置"), 
                             command=self._show_language_dialog)
        
        help_menu.add_separator()
        help_menu.add_command(label=i18n.get_text("about", "关于"), command=self._show_about_dialog)
    
    def _show_language_dialog(self):
        """显示语言设置对话框"""
        dialog = tk.Toplevel(self.master)
        dialog.title(i18n.get_text("dialog_title_language", "语言设置"))
        dialog.geometry("400x300")
        dialog.resizable(True, True)
        dialog.transient(self.master)
        dialog.grab_set()
        
        # 设置对话框样式
        dialog.configure(bg="#1e1e2e")
        
        # 标题标签
        title_label = ttk.Label(
            dialog, 
            text=i18n.get_text("dialog_select_language", "选择界面语言"), 
            style="Title.TLabel", 
            font=("Helvetica", 14, "bold")
        )
        title_label.pack(pady=(20, 20))
        
        # 语言选择框架
        language_frame = ttk.Frame(dialog, style="Card.TFrame", padding=15)
        language_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=0)
        
        # 语言选择
        current_lang = i18n.get_current_language()
        lang_var = tk.StringVar(value=current_lang)
        
        for lang_code, lang_name in i18n.get_all_languages().items():
            lang_radio = ttk.Radiobutton(
                language_frame,
                text=lang_name,
                variable=lang_var,
                value=lang_code,
                style="TRadiobutton"
            )
            lang_radio.pack(anchor=tk.W, pady=5)
        
        # 提示信息
        note_label = ttk.Label(
            dialog,
            text=i18n.get_text("dialog_language_restart_note", 
                             "注意：切换语言后需要重启应用程序以完全生效"),
            style="Card.TLabel",
            wraplength=340
        )
        note_label.pack(pady=15, padx=30)
        
        # 按钮框架
        btn_frame = ttk.Frame(dialog, style="TFrame")
        btn_frame.pack(pady=(0, 20), fill=tk.X)
        
        # 取消按钮
        cancel_btn = ttk.Button(
            btn_frame,
            text=i18n.get_text("btn_cancel", "取消"),
            command=dialog.destroy,
            style="Modern.TButton",
            width=10
        )
        cancel_btn.pack(side=tk.RIGHT, padx=(5, 30))
        
        # 应用按钮
        def apply_language_change():
            new_lang = lang_var.get()
            if new_lang != current_lang:
                if i18n.load_language(new_lang):
                    # 保存语言设置
                    try:
                        settings_dir = os.path.join(os.path.expanduser('~'), '.config', 'yolo_studio')
                        os.makedirs(settings_dir, exist_ok=True)
                        
                        settings_file = os.path.join(settings_dir, 'settings.json')
                        settings = {}
                        
                        if os.path.exists(settings_file):
                            try:
                                with open(settings_file, 'r', encoding='utf-8') as f:
                                    settings = json.load(f)
                            except:
                                pass
                        
                        settings['language'] = new_lang
                        
                        with open(settings_file, 'w', encoding='utf-8') as f:
                            json.dump(settings, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"保存语言设置失败: {e}")
                    
                    # 应用一些可以立即生效的UI文本变更
                    self._refresh_ui_texts()
                    
                    # 显示更清晰的提示，需要重启应用
                    restart_message = i18n.get_text(
                        "language_restart_required",
                        "语言已成功切换为 {language}。\n\n要完全切换界面语言，请重启应用程序。\n部分元素（如菜单）已立即更新。"
                    ).format(language=i18n.get_all_languages()[new_lang])
                    
                    messagebox.showinfo(
                        i18n.get_text("language_settings", "语言设置"),
                        restart_message
                    )
                else:
                    messagebox.showerror(
                        i18n.get_text("language_settings", "语言设置"),
                        i18n.get_text("language_switch_failed", "语言切换失败")
                    )
            dialog.destroy()
        
        apply_btn = ttk.Button(
            btn_frame,
            text=i18n.get_text("btn_apply", "应用"),
            command=apply_language_change,
            style="Primary.TButton",
            width=10
        )
        apply_btn.pack(side=tk.RIGHT, padx=(0, 5))
        
        # 对话框居中显示
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # 设置关闭事件
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    
    def _refresh_ui_texts(self):
        """刷新UI文本以应用新语言设置"""
        try:
            # 更新窗口标题
            if hasattr(self, 'master'):
                self.master.title(i18n.get_text("app_title", "YOLO Studio"))
                
            # 更新菜单
            if hasattr(self, 'menu_bar') and self.menu_bar is not None:
                # 重新创建菜单以更新文本
                self._create_license_menu()
            
            # 更新标签页标签
            if hasattr(self, 'tab_control') and self.tab_control is not None:
                tabs_mapping = {
                    0: "main_tab_annotation",
                    1: "main_tab_training",
                    2: "main_tab_export",
                    3: "main_tab_inference"
                }
                
                for tab_idx, text_key in tabs_mapping.items():
                    try:
                        self.tab_control.tab(tab_idx, text=i18n.get_text(text_key))
                    except:
                        pass
            
            # 更新导航按钮文本
            nav_button_mapping = {
                'annotate_btn': "main_tab_annotation", 
                'train_btn': "main_tab_training",
                'export_btn': "main_tab_export", 
                'infer_btn': "main_tab_inference"
            }
            
            for btn_name, text_key in nav_button_mapping.items():
                if hasattr(self, btn_name) and getattr(self, btn_name) is not None:
                    getattr(self, btn_name).config(text=i18n.get_text(text_key))
                    
            # 更新各页面中的常见按钮
            if self.current_page == "annotate" and hasattr(self, 'annotation_page_frame'):
                # 更新标注页面中的按钮文本
                for child in self.annotation_page_frame.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for widget in child.winfo_children():
                            if isinstance(widget, ttk.Button):
                                text = widget.cget("text")
                                for key in ["btn_save", "btn_open", "btn_cancel", "btn_apply"]:
                                    if text == i18n.get_text(key, text):
                                        widget.config(text=i18n.get_text(key, text))
            
            # 更新状态栏文本
            if hasattr(self, 'status_bar') and self.status_bar:
                current_text = self.status_bar.cget("text")
                status_parts = current_text.split(" - ")
                if len(status_parts) > 1:
                    # 保留许可证状态部分，只更新前缀
                    license_part = " - ".join(status_parts[1:])
                    self.status_bar.config(text=f"{i18n.get_text('app_title', 'YOLO Studio')} - {license_part}")
                else:
                    self.status_bar.config(text=i18n.get_text('status_ready', current_text))
                    
        except Exception as e:
            print(f"刷新UI文本失败: {e}")
            import traceback
            print(traceback.format_exc())

    def _show_about_dialog(self):
        """显示关于对话框"""
        try:
            # 获取完整的许可证信息
            from security import get_license_info
            license_info = get_license_info()
            is_pro = license_info.get("is_pro", False)
            is_trial = license_info.get("is_trial", False)
            expiry_date = license_info.get("expiry_date", "")
            
            # 格式化许可证状态文本
            if is_pro:
                if is_trial:
                    license_text = i18n.get_text("license_pro_trial", "专业版(试用)")
                    if expiry_date:
                        try:
                            # 从ISO格式提取日期部分
                            license_text += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_date.split('T')[0]}"
                        except:
                            license_text += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_date}"
                else:
                    license_text = i18n.get_text("license_pro", "专业版")
                    if expiry_date:
                        try:
                            license_text += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_date.split('T')[0]}"
                        except:
                            license_text += f"\n{i18n.get_text('license_expiry_date', '到期日期')}: {expiry_date}"
            else:
                license_text = i18n.get_text("license_free", "免费版")
        except Exception:
            # 如果获取许可证信息失败，使用简单的版本检查
            license_text = i18n.get_text("license_pro", "专业版") if is_pro_version() else i18n.get_text("license_free", "免费版")
            
        about_title = i18n.get_text("about", "关于") + " YOLO Studio"
        about_message = f"YOLO Studio {license_text}\n"
        about_message += f"{i18n.get_text('version', '版本')}: 1.0.0\n\n"
        about_message += i18n.get_text(
            "about_description", 
            "一款用于YOLOv5/YOLOv8目标检测的全功能工具\n包含数据标注、模型训练和模型导出功能"
        )
        about_message += f"\n\n© 2023-2024 YOLO Studio Team"
        
        messagebox.showinfo(about_title, about_message)
    
    def _add_to_output_log_queue(self, message, is_error=False, is_train=False, is_export=False):
        # 添加消息到输出日志队列
        # is_error标记错误消息
        # is_train标记训练相关消息
        # is_export标记导出相关消息
        self.output_log_queue.append((message, is_error, is_train, is_export))

    def _process_output_log_queue(self):
        # ... (as before)
        # Ensure error tag configuration is robust
        error_tag_configured = False
        if hasattr(self, 'output_text') and self.output_text:
            if "error_tag" in self.output_text.tag_names():
                error_tag_configured = True
            else:
                try: # output_text might not be fully ready on first call
                    self.output_text.tag_configure("error_tag", foreground=config.UI_ERROR_COLOR)
                    error_tag_configured = True
                except tk.TclError:
                    pass # Widget not ready yet
                
        # 配置训练输出文本框的错误标签
        train_error_tag_configured = False
        if hasattr(self, 'output_text_train') and self.output_text_train:
            if "error_tag" in self.output_text_train.tag_names():
                train_error_tag_configured = True
            else:
                try:
                    self.output_text_train.tag_configure("error_tag", foreground=config.UI_ERROR_COLOR)
                    train_error_tag_configured = True
                except tk.TclError:
                    pass
                    
        # 配置导出输出文本框的错误标签
        export_error_tag_configured = False
        if hasattr(self, 'output_text_export') and self.output_text_export:
            if "error_tag" in self.output_text_export.tag_names():
                export_error_tag_configured = True
            else:
                try:
                    self.output_text_export.tag_configure("error_tag", foreground=config.UI_ERROR_COLOR)
                    export_error_tag_configured = True
                except tk.TclError:
                    pass

        while self.output_log_queue:
            message, is_error, is_train, is_export = self.output_log_queue.pop(0)
            
            # 更新通用输出文本框
            if hasattr(self, 'output_text') and self.output_text:
                try:
                    self.output_text.config(state="normal")
                    tag_to_use = "error_tag" if is_error and error_tag_configured else None
                    self.output_text.insert(tk.END, message + "\n", tag_to_use)
                    self.output_text.see(tk.END)
                    self.output_text.config(state="disabled")
                except tk.TclError: # If widget gets destroyed during processing
                    pass
                    
            # 更新训练输出文本框
            if is_train and hasattr(self, 'output_text_train') and self.output_text_train:
                try:
                    self.output_text_train.config(state="normal")
                    tag_to_use = "error_tag" if is_error and train_error_tag_configured else None
                    self.output_text_train.insert(tk.END, message + "\n", tag_to_use)
                    self.output_text_train.see(tk.END)
                    self.output_text_train.config(state="disabled")
                except tk.TclError:
                    pass
            
            # 更新导出输出文本框
            if is_export and hasattr(self, 'output_text_export') and self.output_text_export:
                try:
                    self.output_text_export.config(state="normal")
                    tag_to_use = "error_tag" if is_error and export_error_tag_configured else None
                    self.output_text_export.insert(tk.END, message + "\n", tag_to_use)
                    self.output_text_export.see(tk.END)
                    self.output_text_export.config(state="disabled")
                except tk.TclError:
                    pass
                        
        self.master.after(100, self._process_output_log_queue)

    def _switch_to_page(self, page_name):
        # ... (forget previous frames)
        current_frame_is_mapped = False
        active_frame = None

        if page_name == "annotate" and hasattr(self, 'annotation_page_frame'):
            active_frame = self.annotation_page_frame
        elif page_name == "train" and hasattr(self, 'training_page_frame'):
            active_frame = self.training_page_frame
        elif page_name == "export" and hasattr(self, 'export_page_frame'):
            active_frame = self.export_page_frame
        elif page_name == "inference" and hasattr(self, 'inference_page_frame'):
            active_frame = self.inference_page_frame

        if active_frame and active_frame.winfo_ismapped() and page_name == self.current_page:
            return # Already on the correct, mapped page

        if hasattr(self, 'annotation_page_frame'): self.annotation_page_frame.pack_forget()
        if hasattr(self, 'training_page_frame'): self.training_page_frame.pack_forget()
        if hasattr(self, 'export_page_frame'): self.export_page_frame.pack_forget()
        if hasattr(self, 'inference_page_frame'): self.inference_page_frame.pack_forget()
        # Reset button styles
        self.annotate_btn.configure(style="Nav.TButton")
        self.train_btn.configure(style="Nav.TButton")
        if hasattr(self, 'export_btn'): self.export_btn.configure(style="Nav.TButton")
        if hasattr(self, 'infer_btn'): self.infer_btn.configure(style="Nav.TButton")

        if page_name == "annotate":
            self.annotation_page_frame.pack(fill=tk.BOTH, expand=True)
            self.annotate_btn.configure(style="Accent.TButton")
            self.master.title(f"{i18n.get_text('app_title', 'YOLO Studio')} - {i18n.get_text('main_tab_annotation', '标注页面')}")
            if hasattr(self.annotation_handler, 'redraw_canvas'): self.annotation_handler.redraw_canvas()
        elif page_name == "train":
            self.training_page_frame.pack(fill=tk.BOTH, expand=True)
            self.train_btn.configure(style="Accent.TButton")
            self.master.title(f"{i18n.get_text('app_title', 'YOLO Studio')} - {i18n.get_text('main_tab_training', '训练页面')}")
            if hasattr(self.training_handler, '_update_weights_combo'): self.training_handler._update_weights_combo()
            # 专业版断点恢复按钮可见性
            if hasattr(self, 'resume_train_btn'):
                self.resume_train_btn.config(state="normal" if is_pro_version() else "disabled")
        elif page_name == "export":
            self.export_page_frame.pack(fill=tk.BOTH, expand=True)
            if hasattr(self, 'export_btn'): self.export_btn.configure(style="Accent.TButton")
            self.master.title(f"{i18n.get_text('app_title', 'YOLO Studio')} - {i18n.get_text('main_tab_export', '模型导出页面')}")
            if not self.export_selected_format.get() and list(self.export_handler.export_formats.keys()):
                self.export_selected_format.set(list(self.export_handler.export_formats.keys())[0])
            self.export_handler.on_export_format_selected()
        elif page_name == "inference":
            # 切换推理页面时，始终重建布局，避免界面错乱
            if hasattr(self, 'inference_page_frame') and self.inference_page_frame is not None:
                self.inference_page_frame.destroy()
            ui_components.create_inference_page_layout(self)
            self.inference_page_frame.pack(fill=tk.BOTH, expand=True)
            if hasattr(self, 'infer_btn'):
                self.infer_btn.configure(style="Accent.TButton")
            self.master.title(f"{i18n.get_text('app_title', 'YOLO Studio')} - {i18n.get_text('main_tab_inference', '推理页面')}")
            # 专业版校验
            if not is_pro_version():
                for widget in self.inference_page_frame.winfo_children():
                    widget.grid_forget() if hasattr(widget, 'grid_info') else widget.pack_forget()
                pro_tip = ttk.Label(self.inference_page_frame, text=i18n.get_text("inference_pro_only", "推理功能为专业版专属，请激活专业版后使用。"), style="Title.TLabel", anchor="center")
                pro_tip.grid(row=0, column=0, sticky="nsew")
                self.inference_page_frame.rowconfigure(0, weight=1)
                self.inference_page_frame.columnconfigure(0, weight=1)

        # 解决页面初始空白问题：强制更新布局和绘制
        self.master.update()
        self.master.update_idletasks()
        
        # 主动调用滚动区域的配置更新
        if page_name == "annotate":
            # 找到左侧面板Canvas并更新滚动区域
            for child in self.annotation_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            subchild.configure(scrollregion=subchild.bbox("all"))
                            for item_id in subchild.find_withtag("left_panel_container"):
                                subchild.itemconfig(item_id, width=subchild.winfo_width())
                            break
                    break
        elif page_name == "train":
            # 找到训练页面中的配置面板Canvas并更新滚动区域
            for child in self.training_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            subchild.configure(scrollregion=subchild.bbox("all"))
                            for item_id in subchild.find_withtag("config_panel_container"):
                                subchild.itemconfig(item_id, width=subchild.winfo_width())
                            break
                    break
        elif page_name == "export":
            # 找到导出页面中的设置面板Canvas并更新滚动区域
            for child in self.export_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            subchild.configure(scrollregion=subchild.bbox("all"))
                            for item_id in subchild.find_withtag("settings_panel_container"):
                                subchild.itemconfig(item_id, width=subchild.winfo_width())
                            break
                    break
        elif page_name == "inference":
            # 找到推理页面中的设置面板Canvas并更新滚动区域
            for child in self.inference_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            subchild.configure(scrollregion=subchild.bbox("all"))
                            for item_id in subchild.find_withtag("settings_panel_container"):
                                subchild.itemconfig(item_id, width=subchild.winfo_width())
                            break
                    break

        self.current_page = page_name

    def _bind_events(self):
        # ... (existing bindings as before)
        # Ensure canvas bindings are only active/relevant if canvas exists
        if hasattr(self, 'canvas'):
            self.canvas.bind("<ButtonPress-1>", self.annotation_handler.on_mouse_press)
            self.canvas.bind("<B1-Motion>", self.annotation_handler.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.annotation_handler.on_mouse_release)
            self.canvas.bind("<Motion>", self.annotation_handler.on_mouse_move_canvas)
            self.canvas.bind("<Control-MouseWheel>", self.annotation_handler.on_mouse_wheel_zoom)
            if platform.system() == "Linux":
                self.canvas.bind("<Button-4>", self.annotation_handler.on_mouse_wheel_scroll_vertical_linux)
                self.canvas.bind("<Button-5>", self.annotation_handler.on_mouse_wheel_scroll_vertical_linux)
            else:
                self.canvas.bind("<MouseWheel>", self.annotation_handler.on_mouse_wheel_scroll_vertical)
            self.canvas.bind("<ButtonPress-2>", self.annotation_handler.on_pan_start) 
            self.canvas.bind("<B2-Motion>", self.annotation_handler.on_pan_drag)
            self.canvas.bind("<ButtonRelease-2>", self.annotation_handler.on_pan_end)

        # 绑定窗口大小变化事件
        self.master.bind("<Configure>", self._on_window_resize)
        
        # 绑定全局鼠标移动事件，用于检测鼠标是否在滚动区域上
        self.master.bind("<Motion>", self._detect_active_scroll_area)
        
        # 绑定全局鼠标滚轮事件
        if platform.system() == "Linux":
            self.master.bind_all("<Button-4>", self._on_global_mousewheel)
            self.master.bind_all("<Button-5>", self._on_global_mousewheel)
        else:
            self.master.bind_all("<MouseWheel>", self._on_global_mousewheel)

        self.master.bind_all("<a>", lambda event: self.annotation_handler.prev_image() if self.current_page == "annotate" else None)
        self.master.bind_all("<d>", lambda event: self.annotation_handler.next_image() if self.current_page == "annotate" else None)
        self.master.bind_all("<Delete>", lambda event: self.annotation_handler.delete_selected_annotation_action() if self.current_page == "annotate" else None)
        self.master.bind_all("<Control-s>", lambda event: self.annotation_handler.save_annotations() if self.current_page == "annotate" else None)
        self.master.bind_all("<Control-z>", lambda event: self.annotation_handler.undo_action() if self.current_page == "annotate" else None)
        self.master.bind_all("<Control-y>", lambda event: self.annotation_handler.redo_action() if self.current_page == "annotate" else None)
        # ... (other global key bindings)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _on_window_resize(self, event):
        """处理窗口大小变化事件"""
        # 仅处理主窗口的大小变化事件，忽略子窗口的事件
        if event is None or event.widget == self.master:
            # 更新标注页面的Canvas滚动区域
            if hasattr(self, 'annotation_page_frame') and self.annotation_page_frame.winfo_ismapped():
                # 找到标注页面中的左侧面板Canvas和其窗口
                for child in self.annotation_page_frame.winfo_children():
                    if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                        for subchild in child.winfo_children():
                            if isinstance(subchild, tk.Canvas):
                                # 更新Canvas的滚动区域
                                subchild.configure(scrollregion=subchild.bbox("all"))
                                # 更新窗口宽度，使其与Canvas宽度匹配
                                for item_id in subchild.find_withtag("left_panel_container"):
                                    subchild.itemconfig(item_id, width=subchild.winfo_width())
                                break
                        break
                        
            # 更新训练页面的Canvas滚动区域
            if hasattr(self, 'training_page_frame') and self.training_page_frame.winfo_ismapped():
                # 找到训练页面中的Canvas和其窗口
                for child in self.training_page_frame.winfo_children():
                    if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                        for subchild in child.winfo_children():
                            if isinstance(subchild, tk.Canvas):
                                # 更新Canvas的滚动区域
                                subchild.configure(scrollregion=subchild.bbox("all"))
                                # 更新窗口宽度，使其与Canvas宽度匹配
                                for item_id in subchild.find_withtag("config_panel_container"):
                                    subchild.itemconfig(item_id, width=subchild.winfo_width())
                                break
                        break
            
            # 更新导出页面的Canvas滚动区域
            if hasattr(self, 'export_page_frame') and self.export_page_frame.winfo_ismapped():
                # 找到导出页面中的Canvas和其窗口
                for child in self.export_page_frame.winfo_children():
                    if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                        for subchild in child.winfo_children():
                            if isinstance(subchild, tk.Canvas):
                                # 更新Canvas的滚动区域
                                subchild.configure(scrollregion=subchild.bbox("all"))
                                # 更新窗口宽度，使其与Canvas宽度匹配
                                for item_id in subchild.find_withtag("settings_panel_container"):
                                    subchild.itemconfig(item_id, width=subchild.winfo_width())
                                break
                        break

    def on_closing(self):
        # 保存训练配置
        if hasattr(self, 'training_handler') and hasattr(self.training_handler, 'save_training_settings'):
            try:
                self.training_handler.save_training_settings()
                print("已保存训练配置")
            except Exception as e:
                print(f"保存训练配置失败: {e}")
        
        # 处理训练进程
        if self.is_training:
            if messagebox.askyesno("训练进行中", "训练仍在进行中。确定要退出吗？", parent=self.master):
                self.training_handler.stop_training() 
                self.master.after(500, self.master.destroy) 
        else:
            self.master.destroy()

    def _detect_active_scroll_area(self, event):
        """检测鼠标是否位于某个可滚动区域内"""
        # 根据当前页面检测不同的滚动区域
        scroll_canvas = None
        
        if self.current_page == "annotate":
            # 检查标注页面的左侧面板
            for child in self.annotation_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            # 检查鼠标是否在这个Canvas上
                            x, y, width, height = self._get_widget_bounds(subchild)
                            if x <= event.x_root <= x + width and y <= event.y_root <= y + height:
                                scroll_canvas = subchild
                            break
                    break
        elif self.current_page == "train":
            # 检查训练页面的配置面板
            for child in self.training_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            # 检查鼠标是否在这个Canvas上
                            x, y, width, height = self._get_widget_bounds(subchild)
                            if x <= event.x_root <= x + width and y <= event.y_root <= y + height:
                                scroll_canvas = subchild
                            break
                    break
        elif self.current_page == "export":
            # 检查导出页面的设置面板
            for child in self.export_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            # 检查鼠标是否在这个Canvas上
                            x, y, width, height = self._get_widget_bounds(subchild)
                            if x <= event.x_root <= x + width and y <= event.y_root <= y + height:
                                scroll_canvas = subchild
                            break
                    break
        elif self.current_page == "inference":
            # 检查推理页面的设置面板
            for child in self.inference_page_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child.grid_info().get('column') == 0:
                    for subchild in child.winfo_children():
                        if isinstance(subchild, tk.Canvas):
                            # 检查鼠标是否在这个Canvas上
                            x, y, width, height = self._get_widget_bounds(subchild)
                            if x <= event.x_root <= x + width and y <= event.y_root <= y + height:
                                scroll_canvas = subchild
                            break
                    break
        
        self.active_scroll_canvas = scroll_canvas
    
    def _get_widget_bounds(self, widget):
        """获取控件在屏幕上的坐标和尺寸"""
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        width = widget.winfo_width()
        height = widget.winfo_height()
        return x, y, width, height
    
    def _on_global_mousewheel(self, event):
        """全局鼠标滚轮事件处理"""
        if self.active_scroll_canvas:
            if platform.system() == "Linux":
                if event.num == 4:  # 向上滚动
                    self.active_scroll_canvas.yview_scroll(-1, "units")
                elif event.num == 5:  # 向下滚动
                    self.active_scroll_canvas.yview_scroll(1, "units")
            else:
                self.active_scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _post_init_update(self):
        """初始化后的额外更新，确保所有元素都正确显示"""
        # 强制更新窗口
        self.master.update()
        self.master.update_idletasks()
        
        # 手动触发窗口大小变化事件，重新计算所有滚动区域
        self._on_window_resize(None)
        
        # 确保Canvas滚动区域正确渲染
        if self.current_page == "annotate" and hasattr(self, 'annotation_handler'):
            self.annotation_handler.redraw_canvas()
            
        # 重新更新一次，确保显示正确
        self.master.update_idletasks()

    def _refresh_license_status(self):
        """统一刷新状态栏许可证显示"""
        try:
            from security import get_license_info
            license_info = get_license_info()
            is_pro = license_info.get("is_pro", False)
            is_trial = license_info.get("is_trial", False)
            expiry_date = license_info.get("expiry_date", "")
            expiry_display = ""
            if expiry_date:
                try:
                    expiry_display = expiry_date.split("T")[0]
                except:
                    expiry_display = expiry_date
            if hasattr(self, 'status_bar') and self.status_bar:
                if is_pro:
                    if is_trial:
                        status_text = f"YOLO Studio - 专业版(试用) {expiry_display}"
                    else:
                        status_text = f"YOLO Studio - 专业版 {expiry_display}"
                else:
                    status_text = f"YOLO Studio - 免费版"
                self.status_bar.config(text=status_text)
        except Exception as e:
            self._add_to_output_log_queue(f"刷新许可证状态错误: {str(e)}", is_error=True)

    # 在训练页面设置yolo源码路径时，自动同步到全局变量
    def set_global_yolo_code_path(self, path):
        self.global_yolo_code_path.set(path)
        self.yolo_code_path.set(path)
        self._add_to_output_log_queue(f"已设置全局YOLO源码路径: {path}")

    def _initialize_language_support(self):
        """初始化语言支持"""
        try:
            # 尝试初始化i18n模块
            current_lang = i18n.initialize()
            print(f"初始化默认语言: {current_lang}")
            
            # 检查是否有用户设置的语言选项
            settings_file = os.path.join(os.path.expanduser('~'), '.config', 'yolo_studio', 'settings.json')
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                    
                    if 'language' in settings:
                        # 加载用户设置的语言
                        selected_lang = settings['language']
                        success = i18n.load_language(selected_lang)
                        print(f"加载用户设置语言 {selected_lang}: {'成功' if success else '失败'}")
                        
                        if success:
                            # 初始化时直接更新窗口标题
                            self.master.title(f"{i18n.get_text('app_title', 'YOLO Studio')}")
                            # 立即更新菜单，确保能显示正确的语言
                            self._create_license_menu()
                except Exception as e:
                    print(f"加载用户语言设置失败: {e}")
                    
            # 确保翻译文件存在
            self._ensure_translation_files()
            
        except Exception as e:
            print(f"初始化语言支持失败: {e}")
            
        # 输出当前语言状态，帮助调试
        print(f"当前使用的语言: {i18n.get_current_language()} ({i18n.get_current_language_name()})")
        
    def _ensure_translation_files(self):
        """确保翻译文件存在，如果不存在则创建"""
        try:
            lang_dir = i18n.get_language_dir()
            os.makedirs(lang_dir, exist_ok=True)
            
            # 检查英文翻译文件
            en_file = os.path.join(lang_dir, 'en.json')
            if not os.path.exists(en_file):
                # 创建基本的英文翻译文件
                en_translations = {
                    "app_title": "YOLO Studio",
                    "language": "Language",
                    "help_menu": "Help",
                    "about": "About",
                    "license_management": "License Management",
                    "language_settings": "Language Settings",
                    
                    "main_tab_annotation": "Data Annotation",
                    "main_tab_training": "Model Training",
                    "main_tab_export": "Model Export",
                    "main_tab_inference": "Inference",
                    
                    "btn_ok": "OK",
                    "btn_cancel": "Cancel",
                    "btn_apply": "Apply",
                    "btn_close": "Close",
                    "btn_save": "Save",
                    
                    "dialog_title_language": "Language Settings",
                    "dialog_select_language": "Select Interface Language",
                    "dialog_language_restart_note": "Note: A restart is required for all changes to take effect.",
                    
                    "no_dataset_selected": "No dataset file selected",
                    "yolo_code_path_not_set": "YOLO code path not set",
                    "inference_pro_only": "Inference feature is for Pro version only. Please activate Pro version to use it."
                }
                
                with open(en_file, 'w', encoding='utf-8') as f:
                    json.dump(en_translations, f, ensure_ascii=False, indent=4)
                print(f"Created English translation file at {en_file}")
                
            # 检查中文翻译文件
            zh_file = os.path.join(lang_dir, 'zh_CN.json')
            if not os.path.exists(zh_file):
                # 创建基本的中文翻译文件
                zh_translations = {
                    "app_title": "YOLO Studio",
                    "language": "语言",
                    "help_menu": "帮助",
                    "about": "关于",
                    "license_management": "许可证管理",
                    "language_settings": "语言设置",
                    
                    "main_tab_annotation": "数据标注",
                    "main_tab_training": "模型训练",
                    "main_tab_export": "模型导出",
                    "main_tab_inference": "推理测试",
                    
                    "btn_ok": "确定",
                    "btn_cancel": "取消",
                    "btn_apply": "应用",
                    "btn_close": "关闭",
                    "btn_save": "保存",
                    
                    "dialog_title_language": "语言设置",
                    "dialog_select_language": "选择界面语言",
                    "dialog_language_restart_note": "注意：切换语言后需要重启应用程序以完全生效",
                    
                    "no_dataset_selected": "未选择数据集文件",
                    "yolo_code_path_not_set": "未设置YOLO代码路径",
                    "inference_pro_only": "推理功能为专业版专属，请激活专业版后使用。"
                }
                
                with open(zh_file, 'w', encoding='utf-8') as f:
                    json.dump(zh_translations, f, ensure_ascii=False, indent=4)
                print(f"Created Chinese translation file at {zh_file}")
                
        except Exception as e:
            print(f"Error ensuring translation files: {e}")

# Modified create_navigation_bar to be part of YoloAnnotatorTrainer or called by it
# This is better placed inside ui_components.create_main_ui, 
# but if it needs to be here for some reason, ensure self.export_btn is handled.
# For now, assuming it's correctly handled within ui_components.create_navigation_bar
# that is called by ui_components.create_main_ui.
# We just need to ensure the export_btn is added there.
# A cleaner way is to pass all button configs to create_navigation_bar.

def check_and_install_dependencies():
    # Added onnx and onnxruntime for export feature
    required_packages = {
        'PyYAML': 'yaml', 
        'Pillow': 'PIL', 
        'requests': 'requests',
        'torch': 'torch',               # For PyTorch model loading
        'torchvision': 'torchvision',   # Often needed with torch models
        'onnx': 'onnx',                 # For ONNX export and checking
        'onnxruntime': 'onnxruntime',   # For ONNX validation (optional but good)
        'cryptography': 'cryptography'  # 添加加密库依赖
    }
    missing_packages_pip = []
    # ... (rest of dependency check logic as before)
    try:
        tk.Tk().destroy() 
    except tk.TclError:
        print("错误: Tkinter (GUI library) 未找到或无法初始化。")
        print("请确保您的Python环境已安装Tkinter。 (e.g., sudo apt-get install python3-tk)")
        sys.exit(1)

    print("正在检查核心依赖包...")
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pip_name} ({import_name}) 已安装")
        except ImportError:
            print(f"  ✗ {pip_name} ({import_name}) 未找到，准备安装...")
            missing_packages_pip.append(pip_name)
    
    if missing_packages_pip:
        print(f"\n正在安装缺失的依赖包: {', '.join(missing_packages_pip)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            for pkg_to_install in missing_packages_pip:
                print(f"安装 {pkg_to_install}...")
                # Special handling for torch to potentially get CUDA version if not explicitly handled by user
                if pkg_to_install == 'torch' and 'torchvision' in missing_packages_pip and 'torchaudio' in missing_packages_pip:
                     # This is a basic install, user might need specific CUDA version.
                     # For a robust solution, PyTorch installation should ideally be handled by the
                     # training_handler's install_pytorch_cuda method or a similar dedicated function.
                     # Here, we do a generic install if torch itself is missing.
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
                     # Remove them from list if handled together
                    if 'torchvision' in missing_packages_pip: missing_packages_pip.remove('torchvision')
                    if 'torchaudio' in missing_packages_pip: missing_packages_pip.remove('torchaudio') # If torchaudio was added
                elif pkg_to_install in ['torch', 'torchvision', 'torchaudio'] and 'torch' not in missing_packages_pip:
                    pass # Will be handled by torch or already installed
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_to_install])

            print("\n依赖包安装完成！请重新启动脚本。")
            sys.exit(0) 
        except subprocess.CalledProcessError as e:
            print(f"安装依赖包失败: {e}")
            print(f"请手动安装: pip install {' '.join(missing_packages_pip)}")
            sys.exit(1)
    else:
        print("所有核心依赖包已安装。")
    return True


if __name__ == '__main__':
    # 处理命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='YOLO Studio')
    parser.add_argument('--lang', type=str, choices=['zh_CN', 'en'], 
                       help='强制使用指定语言 (zh_CN=中文, en=英文)')
    args = parser.parse_args()
    
    # 如果指定了语言参数，则将其写入配置文件
    if args.lang:
        try:
            settings_dir = os.path.join(os.path.expanduser('~'), '.config', 'yolo_studio')
            os.makedirs(settings_dir, exist_ok=True)
            settings_file = os.path.join(settings_dir, 'settings.json')
            
            # 读取现有设置或创建新设置
            settings = {}
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                except Exception:
                    pass
            
            # 设置语言并保存
            settings['language'] = args.lang
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
                
            print(f"已强制设置语言为: {args.lang}")
        except Exception as e:
            print(f"设置语言失败: {e}")
    
    # Add project root to sys.path to make `from config import config` etc. work reliably
    # This assumes main_app.py is in the project root (e.g., YOLO_Studio).
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


    if check_and_install_dependencies():
        root = tk.Tk()
        app = YoloAnnotatorTrainer(root) # app gets assigned here
        
        # After app is initialized, ui_components.create_main_ui would have created the nav buttons
        # We need to add the export button to the nav_frame created there.
        # This is a bit of a workaround for how create_main_ui is structured.
        # A cleaner way is to pass all button configs to create_navigation_bar.
        
        # Add Export button to the navigation bar (created within ui_components.create_main_ui)
        # Find nav_frame (assuming it's the first child of app.main_container if create_main_ui was called)
        nav_frame_children = app.main_container.winfo_children()
        if nav_frame_children:
            nav_frame = nav_frame_children[0] # Usually the nav_frame


        root.mainloop()