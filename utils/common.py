import sys

def is_running_packaged():
    """检查应用程序是否以打包形式运行"""
    return hasattr(sys, '_MEIPASS') 