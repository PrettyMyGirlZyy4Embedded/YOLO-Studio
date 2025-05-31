# config.py
import sys # <--- 确保存在
import os  

# --- App Configuration ---
APP_VERSION = "1.0.0"

# --- Language Configuration ---
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh_CN": "简体中文"
}
DEFAULT_LANGUAGE = "zh_CN"

# --- UI Theme Configuration ---
UI_BACKGROUND_COLOR = "white"
UI_FOREGROUND_COLOR = "black"
UI_SELECT_BACKGROUND_COLOR = "#e0e0e0"
UI_BUTTON_BACKGROUND_COLOR = "#f0f0f0"
UI_BUTTON_ACTIVE_BACKGROUND_COLOR = "#d0d0d0"
UI_ACCENT_COLOR = "#4CAF50" # Green
UI_WARNING_COLOR = "#FFC107" # Amber
UI_ERROR_COLOR = "#F44336" # Red

# 新增界面颜色配置
BG_PRIMARY = "#ffffff"      # 主背景色
BG_SECONDARY = "#f5f5f5"    # 次要背景色
BG_CARD = "#ffffff"         # 卡片背景色
BG_HOVER = "#e8e8e8"        # 悬停背景色
FG_PRIMARY = "#000000"      # 主文本色
FG_SECONDARY = "#666666"    # 次要文本色
TEXT_PRIMARY = "#000000"    # 主文本色（滚动条箭头等）
TEXT_SECONDARY = "#666666"  # 次要文本色

# --- Annotation Configuration ---
DEFAULT_CLASS_DEFINITIONS = [
    {"name": "person", "color": "red"}, {"name": "car", "color": "blue"},
    {"name": "cat", "color": "green"}, {"name": "dog", "color": "orange"},
    {"name": "bicycle", "color": "purple"}
]
FALLBACK_COLORS = ["#FF69B4", "#00CED1", "#FFD700", "#32CD32", "#8A2BE2",
                   "#FF7F50", "#6495ED", "#DC143C", "#00FFFF", "#ADFF2F"]
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif')
SELECTED_ANNOTATION_COLOR = "gold"
ANNOTATION_TEXT_COLOR = "black"
MIN_SCALE = 0.1
MAX_SCALE = 10.0
MAX_UNDO_HISTORY = 50
HANDLE_SIZE = 4 # Retained in case full resize logic is re-added
HANDLE_COLOR = "black" # Retained
HANDLE_TAG = "resize_handle" # Retained
HANDLE_CLICK_RADIUS_SQUARED = (HANDLE_SIZE * 2.5)**2 # Retained
BORDER_CLICK_TOLERANCE = 5 # Retained for potential future use
MIN_ANNOTATION_SIZE_ON_CANVAS = 5
SCROLL_STEP_VERTICAL = 30

# --- Training Configuration ---
import sys
import os

YOLO_VERSIONS = {
    "YOLOv5": {
        "repo_url_template": "https://github.com/ultralytics/yolov5/archive/refs/tags/{version}.zip",
        "versions": ["v7.0", "v6.2", "v6.1", "v6.0", "v5.0"],
        "requirements": "requirements.txt",
        "train_script_rel_path": "train.py",
        "default_weights": {
            "yolov5n.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
            "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
            "yolov5m.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
            "yolov5l.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
            "yolov5x.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt"
        },
        "command_generator": lambda params: [
            sys.executable, params["train_script_abs_path"],
            "--img", str(params["imgsz"]),
            "--batch", str(params["batch_size"]),
            "--epochs", str(params["epochs"]),
            "--data", params["datasets_yaml_path"],
            "--weights", params["weights_path"],
            "--device", params["device"],
            "--workers", str(params["workers"]),
            "--project", params["project_dir_abs_path"],
            "--name", params["run_name"],
            "--hyp", os.path.join(params["yolo_code_path"], 'data', 'hyps', 'hyp.scratch-low.yaml')
        ]
    },
    "YOLOv8": {
        "repo_url_template": "https://github.com/ultralytics/ultralytics/archive/refs/tags/v{version}.zip",
        "versions": ["8.2.28", "8.1.0", "8.0.196"], # Example versions
        "requirements": "requirements.txt", # Often in ultralytics/requirements.txt
        "train_via_module": "ultralytics",
        "default_weights": {
            "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
            "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
            "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
            "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt"
        },
        "command_generator": lambda params: [
            sys.executable, "-m", "ultralytics", "train",
            f"data={params['datasets_yaml_path']}",
            f"model={params['weights_path']}",
            f"epochs={params['epochs']}",
            f"batch={params['batch_size']}",
            f"imgsz={params['imgsz']}",
            f"device={params['device']}",
            f"workers={params['workers']}",
            f"project={params['project_dir_abs_path']}",
            f"name={params['run_name']}"
        ]
    }
}
DEFAULT_PROJECT_DIR = "yolo_runs"

# --- Export Configuration ---
DEFAULT_EXPORT_DIR = "exported_models"
SUPPORTED_EXPORT_FORMATS = [
    "PyTorch (.pt) -> ONNX",
    "ONNX -> TensorFlow Lite (.tflite)",
    "ONNX -> OpenVINO IR (.xml/.bin)"
]

# --- License Configuration ---
LICENSE_PURCHASE_URL = "https://www.example.com/yolo-studio-purchase"  # 许可证购买页面URL
LICENSE_HELP_URL = "https://www.example.com/yolo-studio-help"  # 许可证帮助页面URL
LICENSE_DIALOG_SIZE = "600x550"  # 许可证对话框大小
LICENSE_PRO_FEATURES = [
    "更多导出格式支持（TensorRT, CoreML等）",
    "批量数据处理和标注",
    "高级模型优化选项",
    "优先技术支持",
    "无水印导出",
    "更多预训练模型"
]  # 专业版特性列表