"""
语言国际化支持模块
提供应用程序多语言支持的核心功能
"""
import os
import json
import logging
import sys

# 导入配置
try:
    from config.config import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
except ImportError:
    # 如果无法导入，使用默认配置
    DEFAULT_LANGUAGE = "zh_CN"
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "zh_CN": "简体中文"
    }

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
_current_language = DEFAULT_LANGUAGE
_translations = {}

def get_language_dir():
    """获取语言文件目录"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lang_dir = os.path.join(base_dir, "resources", "languages")
    # 确保目录存在
    os.makedirs(lang_dir, exist_ok=True)
    logger.info(f"语言文件目录: {lang_dir}")
    return lang_dir

def load_language(lang_code):
    """
    加载指定语言代码的翻译文件
    
    Args:
        lang_code (str): 语言代码，如'en', 'zh_CN'
        
    Returns:
        bool: 加载成功返回True，否则返回False
    """
    global _current_language, _translations
    
    if lang_code not in SUPPORTED_LANGUAGES:
        logger.error(f"不支持的语言代码: {lang_code}")
        return False
    
    lang_dir = get_language_dir()
    lang_file = os.path.join(lang_dir, f"{lang_code}.json")
    
    # 检查语言文件是否存在
    if not os.path.exists(lang_file):
        logger.error(f"语言文件不存在: {lang_file}")
        # 尝试创建一个空的语言文件
        try:
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info(f"创建了空的语言文件: {lang_file}")
        except Exception as e:
            logger.error(f"创建语言文件失败: {e}")
        return False
        
    try:
        with open(lang_file, 'r', encoding='utf-8') as f:
            _translations = json.load(f)
        _current_language = lang_code
        logger.info(f"已加载语言: {SUPPORTED_LANGUAGES[lang_code]}，含有{len(_translations)}个翻译条目")
        return True
    except Exception as e:
        logger.error(f"加载语言文件失败: {e}")
        return False

def get_text(key, default=None):
    """
    获取翻译文本
    
    Args:
        key (str): 翻译键名
        default (str, optional): 未找到翻译时的默认值
        
    Returns:
        str: 翻译文本或默认值
    """
    if key in _translations:
        return _translations[key]
    
    if default is not None:
        return default
    
    # 如果没有找到翻译，返回键名并记录
    # logger.warning(f"未找到翻译键: {key}")
    return key

def get_current_language():
    """获取当前语言代码"""
    return _current_language

def get_current_language_name():
    """获取当前语言名称"""
    return SUPPORTED_LANGUAGES.get(_current_language, "Unknown")

def get_all_languages():
    """获取所有支持的语言列表"""
    return SUPPORTED_LANGUAGES

def debug_info():
    """返回调试信息"""
    info = {
        "current_language": _current_language,
        "language_name": get_current_language_name(),
        "translations_count": len(_translations),
        "language_dir": get_language_dir(),
        "language_files": []
    }
    
    # 检查语言文件
    lang_dir = get_language_dir()
    for lang_code in SUPPORTED_LANGUAGES:
        lang_file = os.path.join(lang_dir, f"{lang_code}.json")
        info["language_files"].append({
            "code": lang_code,
            "name": SUPPORTED_LANGUAGES[lang_code],
            "file_exists": os.path.exists(lang_file),
            "file_path": lang_file
        })
    
    return info

def initialize():
    """初始化语言模块"""
    global _current_language
    
    # 创建语言文件目录（如果不存在）
    lang_dir = get_language_dir()
    
    # 检查命令行参数
    lang_from_args = None
    if "--lang" in sys.argv:
        try:
            idx = sys.argv.index("--lang")
            if idx + 1 < len(sys.argv):
                lang_from_args = sys.argv[idx + 1]
        except:
            pass
    
    # 优先使用命令行参数
    if lang_from_args and lang_from_args in SUPPORTED_LANGUAGES:
        logger.info(f"从命令行参数加载语言: {lang_from_args}")
        if load_language(lang_from_args):
            _current_language = lang_from_args
            return _current_language
    
    # 尝试加载设置中的语言
    try:
        settings_file = os.path.join(os.path.expanduser('~'), '.config', 'yolo_studio', 'settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            if 'language' in settings and settings['language'] in SUPPORTED_LANGUAGES:
                logger.info(f"从设置文件加载语言: {settings['language']}")
                if load_language(settings['language']):
                    _current_language = settings['language']
                    return _current_language
    except Exception as e:
        logger.error(f"从设置文件加载语言失败: {e}")
    
    # 加载默认语言
    logger.info(f"加载默认语言: {DEFAULT_LANGUAGE}")
    if load_language(DEFAULT_LANGUAGE):
        _current_language = DEFAULT_LANGUAGE
    else:
        # 如果默认语言加载失败，尝试加载英文
        logger.info("默认语言加载失败，尝试加载英文")
        if load_language("en"):
            _current_language = "en"
    
    return _current_language 