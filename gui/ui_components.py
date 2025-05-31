# ui_components.py - 优化版本
import tkinter as tk
from tkinter import ttk
import config.config as config
import platform

# 导入i18n模块
try:
    from utils import i18n
except ImportError:
    # 如果国际化模块不存在，使用虚拟实现
    class i18n_mock:
        @staticmethod
        def get_text(key, default=None):
            return default if default is not None else key
        
        @staticmethod
        def get_current_language():
            return "zh_CN"
    
    i18n = i18n_mock()

def configure_styles(style):
    # 现代化配色方案 - 更柔和的深色主题
    bg_primary = "#1e1e2e"     # 深色主背景，更加柔和
    bg_secondary = "#313244"   # 次要背景，微妙对比
    bg_card = "#45475a"        # 卡片背景，更加突出
    text_primary = "#cdd6f4"   # 主文本，不再是纯白色，减少视觉疲劳
    text_secondary = "#a6adc8"  # 次要文本
    accent_blue = "#89b4fa"    # 更加柔和的蓝色
    accent_green = "#a6e3a1"   # 柔和的绿色
    accent_orange = "#fab387"  # 温暖的橙色
    accent_red = "#f38ba8"     # 柔和的红色

    # 基础样式
    style.configure("TFrame", background=bg_primary)
    style.configure("TLabel", background=bg_primary, foreground=text_primary, font=("Microsoft YaHei UI", 9))
    style.configure("Title.TLabel", background=bg_primary, foreground=text_primary, font=("Microsoft YaHei UI", 16, "bold"))
    style.configure("Subtitle.TLabel", background=bg_primary, foreground=text_secondary, font=("Microsoft YaHei UI", 10))

    # 卡片样式
    style.configure("Card.TFrame", background=bg_card, relief="flat", borderwidth=1)
    style.configure("Card.TLabel", background=bg_card, foreground=text_primary, font=("Microsoft YaHei UI", 9))
    style.configure("CardTitle.TLabel", background=bg_card, foreground=text_primary, font=("Microsoft YaHei UI", 11, "bold"))

    # 按钮样式 - 现代扁平设计
    style.configure("Modern.TButton",
                    background=bg_secondary,
                    foreground=text_primary,
                    borderwidth=0,
                    focuscolor='none',
                    padding=(12, 8),
                    font=("Microsoft YaHei UI", 9))
    style.map("Modern.TButton",
              background=[('active', '#585b70'), ('pressed', '#6c7086')])

    # 主要操作按钮
    style.configure("Primary.TButton",
                    background=accent_blue,
                    foreground="#1e1e2e",
                    borderwidth=0,
                    focuscolor='none',
                    padding=(15, 10),
                    font=("Microsoft YaHei UI", 9, "bold"))
    style.map("Primary.TButton",
              background=[('active', '#74c7ec'), ('pressed', '#89dceb')])

    # 成功按钮
    style.configure("Success.TButton",
                    background=accent_green,
                    foreground="#1e1e2e",
                    borderwidth=0,
                    focuscolor='none',
                    padding=(12, 8),
                    font=("Microsoft YaHei UI", 9))
    style.map("Success.TButton",
              background=[('active', '#94e2d5'), ('pressed', '#a6e3a1')])

    # 警告按钮
    style.configure("Warning.TButton",
                    background=accent_orange,
                    foreground="#1e1e2e",
                    borderwidth=0,
                    focuscolor='none',
                    padding=(12, 8),
                    font=("Microsoft YaHei UI", 9))
    style.map("Warning.TButton",
              background=[('active', '#f9e2af'), ('pressed', '#fab387')])

    # 危险按钮
    style.configure("Danger.TButton",
                    background=accent_red,
                    foreground="#1e1e2e",
                    borderwidth=0,
                    focuscolor='none',
                    padding=(12, 8),
                    font=("Microsoft YaHei UI", 9))
    style.map("Danger.TButton",
              background=[('active', '#f38ba8'), ('pressed', '#f5c2e7')])

    # 导航按钮
    style.configure("Nav.TButton",
                    background=bg_secondary,
                    foreground=text_primary,
                    borderwidth=0,
                    focuscolor='none',
                    padding=(20, 12),
                    font=("Microsoft YaHei UI", 10))
    style.map("Nav.TButton",
              background=[('active', '#585b70'), ('pressed', '#6c7086')])

    style.configure("Accent.TButton",
                    background=accent_blue,
                    foreground="#1e1e2e",
                    borderwidth=0,
                    focuscolor='none',
                    padding=(20, 12),
                    font=("Microsoft YaHei UI", 10, "bold"))

    # 输入控件样式
    style.configure("Modern.TEntry",
                    fieldbackground=bg_secondary,
                    background=bg_secondary,
                    foreground=text_primary,
                    borderwidth=1,
                    relief="solid",
                    insertcolor=text_primary,
                    padding=(8, 8),
                    font=("Microsoft YaHei UI", 9))

    style.configure("Modern.TCombobox",
                    fieldbackground=bg_secondary,
                    background=bg_secondary,
                    foreground=text_primary,
                    borderwidth=1,
                    arrowcolor=text_primary,
                    padding=(8, 8),
                    font=("Microsoft YaHei UI", 9))

    # LabelFrame样式
    style.configure("Modern.TLabelframe",
                    background=bg_card,
                    borderwidth=1,
                    relief="solid",
                    bordercolor="#6c7086")
    style.configure("Modern.TLabelframe.Label",
                    background=bg_card,
                    foreground=text_primary,
                    font=("Microsoft YaHei UI", 10, "bold"))

    # 进度条样式
    style.configure("Modern.Horizontal.TProgressbar",
                    background=accent_blue,
                    troughcolor=bg_secondary,
                    borderwidth=0,
                    lightcolor=accent_blue,
                    darkcolor="#89dceb")

def create_section(parent, title_text, card_style="Card.TFrame"):
    """辅助函数：创建一个带标题的区块 (LabelFrame)"""
    section_frame = ttk.LabelFrame(parent, text=title_text, style="Modern.TLabelframe")
    section_frame.pack(fill=tk.X, pady=(10, 15), padx=5)
    # 内部内容框架，使用卡片样式和内边距
    content_frame = ttk.Frame(section_frame, style=card_style, padding=15)
    content_frame.pack(fill=tk.BOTH, expand=True)
    return content_frame


def create_main_ui(app):
    # 主容器
    app.main_container = ttk.Frame(app.master, style="TFrame")
    app.main_container.pack(fill=tk.BOTH, expand=True)

    # 配置 main_container 的 grid 行列权重
    app.main_container.rowconfigure(1, weight=1) # 让页面容器行扩展
    app.main_container.columnconfigure(0, weight=1) # 让所有列内容扩展

    # 创建顶部导航栏
    create_modern_navigation(app)

    # 页面容器 - 使用圆角边框
    app.page_container = ttk.Frame(app.main_container, style="TFrame")
    app.page_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(5, 20))

    # 创建各个页面
    create_annotation_page_layout(app)
    create_training_page_layout(app)
    create_export_page_layout(app)
    create_inference_page_layout(app)

    # 现代化状态栏
    create_modern_status_bar(app)

def create_modern_navigation(app):
    """创建现代化导航栏"""
    nav_container = ttk.Frame(app.main_container, style="TFrame", padding=(0,0,0,0))
    nav_container.grid(row=0, column=0, sticky="ew", padx=20, pady=(20,10)) # 增加顶部间距
    nav_container.columnconfigure(0, weight=0) # 应用标题部分不扩展
    nav_container.columnconfigure(1, weight=1) # 空白区域扩展，使得按钮能靠右
    nav_container.columnconfigure(2, weight=0) # 导航按钮部分不扩展

    # 应用标题
    title_frame = ttk.Frame(nav_container, style="TFrame")
    title_frame.grid(row=0, column=0, sticky="w")

    app_title = ttk.Label(title_frame, text=i18n.get_text("app_title", "YOLO Studio"), style="Title.TLabel")
    app_title.pack(side=tk.LEFT)

    subtitle = ttk.Label(title_frame, text=i18n.get_text("app_subtitle", "专业的目标检测训练平台"), style="Subtitle.TLabel")
    subtitle.pack(side=tk.LEFT, padx=(15, 0))

    # 导航按钮 - 使用更现代的设计
    nav_buttons_frame = ttk.Frame(nav_container, style="TFrame")
    nav_buttons_frame.grid(row=0, column=2, sticky="e")
    
    # 创建导航按钮组，使用平铺设计
    nav_btn_style = {"width": 16, "padding": (15, 10), "font": ("Microsoft YaHei UI", 10)}
    
    app.annotate_btn = ttk.Button(nav_buttons_frame, text="📝 " + i18n.get_text("main_tab_annotation", "数据标注"),
                                  command=lambda: app._switch_to_page("annotate"),
                                  style="Nav.TButton", width=nav_btn_style["width"])
    app.annotate_btn.grid(row=0, column=0, padx=(0,1), ipadx=nav_btn_style["padding"][0], ipady=nav_btn_style["padding"][1])

    app.train_btn = ttk.Button(nav_buttons_frame, text="🚀 " + i18n.get_text("main_tab_training", "模型训练"),
                                 command=lambda: app._switch_to_page("train"),
                                 style="Nav.TButton", width=nav_btn_style["width"])
    app.train_btn.grid(row=0, column=1, padx=(0,1), ipadx=nav_btn_style["padding"][0], ipady=nav_btn_style["padding"][1])

    app.export_btn = ttk.Button(nav_buttons_frame, text="📦 " + i18n.get_text("main_tab_export", "模型导出"),
                                  command=lambda: app._switch_to_page("export"),
                                  style="Nav.TButton", width=nav_btn_style["width"])
    app.export_btn.grid(row=0, column=2, padx=(0,0), ipadx=nav_btn_style["padding"][0], ipady=nav_btn_style["padding"][1])

    app.infer_btn = ttk.Button(nav_buttons_frame, text="🤖 " + i18n.get_text("main_tab_inference", "模型推理"),
                                  command=lambda: app._switch_to_page("inference"),
                                  style="Nav.TButton", width=nav_btn_style["width"])
    app.infer_btn.grid(row=0, column=3, padx=(0,0), ipadx=nav_btn_style["padding"][0], ipady=nav_btn_style["padding"][1])

def create_modern_status_bar(app):
    """创建现代化状态栏"""
    status_frame = ttk.Frame(app.master, style="Card.TFrame", padding=(15, 8))
    status_frame.pack(side=tk.BOTTOM, fill=tk.X)

    app.status_bar = ttk.Label(status_frame, text="🟢 " + i18n.get_text("status_ready", "系统就绪"), style="Card.TLabel")
    app.status_bar.pack(side=tk.LEFT)

    # 版本信息和版权信息
    version_label = ttk.Label(status_frame, text=f"v{config.APP_VERSION} | YOLO Studio © 2025", style="Card.TLabel")
    version_label.pack(side=tk.RIGHT)

def create_annotation_page_layout(app):
    """创建标注页面布局"""
    app.annotation_page_frame = ttk.Frame(app.page_container, style="TFrame")
    # app.annotation_page_frame 在 _switch_to_page 时 pack(fill=tk.BOTH, expand=True)

    # 配置 annotation_page_frame 的列权重
    app.annotation_page_frame.columnconfigure(0, weight=0)  # 左侧面板固定或按内容决定宽度
    app.annotation_page_frame.columnconfigure(1, weight=1)  # 画布区域列扩展
    app.annotation_page_frame.rowconfigure(0, weight=1)     # 整行扩展

    # 左侧控制面板 - 使用带滚动条的Canvas实现内容超出时可滚动
    left_panel_outer = ttk.Frame(app.annotation_page_frame, style="TFrame", width=380) 
    left_panel_outer.grid(row=0, column=0, sticky="ns", padx=(0,15))
    left_panel_outer.grid_propagate(False)  # 固定宽度
    left_panel_outer.rowconfigure(0, weight=1)
    left_panel_outer.columnconfigure(0, weight=1)

    # 创建Canvas和滚动条
    left_canvas = tk.Canvas(left_panel_outer, bg="#1e1e2e", 
                           highlightthickness=0, bd=0)
    left_canvas.grid(row=0, column=0, sticky="nsew")
    
    left_scrollbar = ttk.Scrollbar(left_panel_outer, orient="vertical", 
                                 command=left_canvas.yview)
    left_scrollbar.grid(row=0, column=1, sticky="ns")
    
    left_canvas.configure(yscrollcommand=left_scrollbar.set)
    
    # 内部frame放置实际内容
    left_panel_container = ttk.Frame(left_canvas, style="TFrame")
    left_canvas_window = left_canvas.create_window((0, 0), window=left_panel_container, 
                                                anchor="nw", tags="left_panel_container")
    
    # 实际内容的容器frame
    left_panel = ttk.Frame(left_panel_container, style="Card.TFrame", padding=15)
    left_panel.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # 右侧画布区域 - 添加标题
    app.canvas_frame = ttk.Frame(app.annotation_page_frame, style="Card.TFrame", padding=15)
    app.canvas_frame.grid(row=0, column=1, sticky="nsew")
    app.canvas_frame.rowconfigure(1, weight=1)    # 画布行扩展
    app.canvas_frame.columnconfigure(0, weight=1) # 画布列扩展

    # 添加画布标题
    canvas_title = ttk.Label(app.canvas_frame, text="📸 " + i18n.get_text("image_preview_area", "图像预览与标注区域"), style="CardTitle.TLabel")
    canvas_title.grid(row=0, column=0, sticky="w", pady=(0, 10))

    # === 左侧面板内容 ===
    # 文件操作区
    file_section_content = create_section(left_panel, "📁 " + i18n.get_text("file_operations", "文件操作"))

    open_dir_btn = ttk.Button(file_section_content, text=i18n.get_text("open_image_folder", "打开图片文件夹"), 
                              command=app.annotation_handler.load_image_dir, style="Primary.TButton")
    open_dir_btn.pack(fill=tk.X, pady=(0,10))

    app.current_image_label = ttk.Label(file_section_content, text=i18n.get_text("no_images_loaded", "未加载图片"), 
                                        style="Card.TLabel", anchor="w", wraplength=300)
    app.current_image_label.pack(fill=tk.X, pady=5)

    nav_buttons = ttk.Frame(file_section_content, style="Card.TFrame")
    nav_buttons.pack(fill=tk.X, pady=5)
    nav_buttons.columnconfigure(0, weight=1)
    nav_buttons.columnconfigure(1, weight=1)

    prev_btn = ttk.Button(nav_buttons, text="⬅ " + i18n.get_text("previous_image", "上一张") + " (A)", 
                          command=app.annotation_handler.prev_image, style="Modern.TButton")
    prev_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0,2))

    next_btn = ttk.Button(nav_buttons, text=i18n.get_text("next_image", "下一张") + " (D) ➡", 
                          command=app.annotation_handler.next_image, style="Modern.TButton")
    next_btn.grid(row=0, column=1, sticky=tk.EW, padx=(2,0))

    # 类别控制区
    class_section_content = create_section(left_panel, "🏷️ " + i18n.get_text("class_control", "类别控制"))
    
    class_select_frame = ttk.Frame(class_section_content, style="Card.TFrame")
    class_select_frame.pack(fill=tk.X)
    class_select_frame.columnconfigure(0, weight=1)
    class_select_frame.columnconfigure(1, weight=0)

    app.class_var = tk.StringVar()
    app.class_combobox = ttk.Combobox(class_select_frame, textvariable=app.class_var, state="readonly", style="Modern.TCombobox")
    app.class_combobox.grid(row=0, column=0, sticky=tk.EW, pady=(0,5), padx=(0,5))
    # app.class_combobox.bind("<<ComboboxSelected>>", app.annotation_handler.on_class_select)

    manage_classes_btn = ttk.Button(class_select_frame, text=i18n.get_text("manage", "管理"), 
                                    command=app.annotation_handler.manage_classes, style="Modern.TButton")
    manage_classes_btn.grid(row=0, column=1, pady=(0,5))

    # 标注列表区
    list_section_content = create_section(left_panel, "☰ " + i18n.get_text("annotations_list", "标注列表"))
    
    # 更现代的列表框样式
    app.annotation_listbox = tk.Listbox(list_section_content, height=10, 
                                        background="#313244", 
                                        foreground="#cdd6f4", 
                                        selectbackground="#89b4fa",
                                        selectforeground="#1e1e2e",
                                        borderwidth=0, highlightthickness=0, 
                                        font=("Microsoft YaHei UI", 9),
                                        exportselection=False)
    app.annotation_listbox.pack(fill=tk.BOTH, expand=True, pady=(0,5))
    app.annotation_listbox.bind("<<ListboxSelect>>", app.annotation_handler.on_listbox_select)

    delete_btn = ttk.Button(list_section_content, text=i18n.get_text("delete_selected", "删除选中") + " (Del)", 
                            command=app.annotation_handler.delete_selected_annotation_action, style="Danger.TButton")
    delete_btn.pack(fill=tk.X)

    # 快捷操作区
    actions_section_content = create_section(left_panel, "⚡ " + i18n.get_text("quick_actions", "快捷操作"))
    
    action_buttons = ttk.Frame(actions_section_content, style="Card.TFrame")
    action_buttons.pack(fill=tk.X)
    action_buttons.columnconfigure(0, weight=1)
    action_buttons.columnconfigure(1, weight=1)
    
    save_btn = ttk.Button(action_buttons, text="💾 " + i18n.get_text("save_annotations", "保存标注"), 
                        command=app.annotation_handler.save_annotations, style="Success.TButton")
    save_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,5))
    
    undo_btn = ttk.Button(action_buttons, text="↩ " + i18n.get_text("undo", "撤销"), 
                        command=app.annotation_handler.undo_action, style="Modern.TButton")
    undo_btn.grid(row=1, column=0, sticky="ew", padx=(0,2), pady=(0,5))

    redo_btn = ttk.Button(action_buttons, text="↪ " + i18n.get_text("redo", "重做"), 
                        command=app.annotation_handler.redo_action, style="Modern.TButton")
    redo_btn.grid(row=1, column=1, sticky="ew", padx=(2,0), pady=(0,5))
    
    # 添加快捷键提示
    shortcuts = i18n.get_text(
        "shortcuts_hint", 
        "快捷键: Del-删除 | Ctrl+S-保存 | Ctrl+Z-撤销 | Ctrl+Y-重做"
    )
    shortcut_label = ttk.Label(action_buttons, text=shortcuts, 
                               style="Card.TLabel", wraplength=300, 
                               font=("Microsoft YaHei UI", 8))
    shortcut_label.grid(row=2, column=0, columnspan=2, sticky="ew")

    # === 右侧画布 ===
    canvas_container = ttk.Frame(app.canvas_frame, style="Card.TFrame")
    canvas_container.grid(row=1, column=0, sticky="nsew")
    canvas_container.rowconfigure(0, weight=1)
    canvas_container.columnconfigure(0, weight=1)
    
    app.canvas = tk.Canvas(canvas_container, bg="#313244", highlightthickness=0)
    app.canvas.grid(row=0, column=0, sticky="nsew")

    # 初始绘制
    app.annotation_handler.redraw_canvas()
    
    # 为左侧面板的滚动功能添加绑定
    def _on_left_canvas_configure(event):
        left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left_canvas.itemconfig(left_canvas_window, width=event.width)
    
    left_panel_container.bind("<Configure>", _on_left_canvas_configure)
    
    # 添加鼠标滚轮绑定
    def _on_mousewheel(event):
        if platform.system() == "Windows":
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif platform.system() == "Darwin":  # macOS
            left_canvas.yview_scroll(int(-1*event.delta), "units")
    
    # 为Linux添加滚轮事件
    def _on_mousewheel_linux(event):
        if event.num == 4:  # 向上滚动
            left_canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # 向下滚动
            left_canvas.yview_scroll(1, "units")
    
    # 绑定滚轮事件
    if platform.system() == "Linux":
        left_canvas.bind("<Button-4>", _on_mousewheel_linux)
        left_canvas.bind("<Button-5>", _on_mousewheel_linux)
    else:
        left_canvas.bind("<MouseWheel>", _on_mousewheel)

def create_output_log(app, parent_frame):
    """创建输出日志区域，现在作为参数传入的父框架的子项"""
    log_frame = ttk.LabelFrame(parent_frame, text="输出日志", style="Modern.TLabelframe")
    log_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5)) # 在父容器中应能扩展
    # log_frame.grid(row=X, column=Y, sticky="nsew", pady=(0,5)) # 如果父容器使用grid
    # 需要父容器配置好行列权重

    app.output_text = tk.Text(log_frame, height=8, wrap=tk.WORD, state=tk.DISABLED,
                                bg=config.BG_SECONDARY, fg=config.FG_PRIMARY,
                                relief=tk.SOLID, borderwidth=0, font=("Segoe UI", 9),
                                selectbackground=config.UI_ACCENT_COLOR, selectforeground="white")
    app.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    scrollbar = ttk.Scrollbar(log_frame, command=app.output_text.yview, style="Modern.Vertical.TScrollbar")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app.output_text.config(yscrollcommand=scrollbar.set)

def create_training_page_layout(app):
    """创建训练页面布局"""
    app.training_page_frame = ttk.Frame(app.page_container, style="TFrame")
    # app.training_page_frame 在 _switch_to_page 时 pack(fill=tk.BOTH, expand=True)

    # 配置 training_page_frame 的列权重
    app.training_page_frame.columnconfigure(0, weight=0)  # 左侧配置面板
    app.training_page_frame.columnconfigure(1, weight=1)  # 右侧输出面板扩展
    app.training_page_frame.rowconfigure(0, weight=1)     # 整行扩展

    # 左侧配置面板 - 使用带滚动条的Canvas实现内容超出时可滚动
    config_panel_outer = ttk.Frame(app.training_page_frame, style="TFrame", width=430) 
    config_panel_outer.grid(row=0, column=0, sticky="ns", padx=(0,15))
    config_panel_outer.grid_propagate(False)  # 固定宽度
    config_panel_outer.rowconfigure(0, weight=1)
    config_panel_outer.columnconfigure(0, weight=1)

    # 创建Canvas和滚动条
    config_canvas = tk.Canvas(config_panel_outer, bg=config.UI_BACKGROUND_COLOR, 
                              highlightthickness=0, bd=0)
    config_canvas.grid(row=0, column=0, sticky="nsew")
    
    config_scrollbar = ttk.Scrollbar(config_panel_outer, orient="vertical", 
                                    command=config_canvas.yview)
    config_scrollbar.grid(row=0, column=1, sticky="ns")
    
    config_canvas.configure(yscrollcommand=config_scrollbar.set)
    
    # 内部frame放置实际内容
    config_panel_container = ttk.Frame(config_canvas, style="TFrame")
    config_canvas_window = config_canvas.create_window((0, 0), window=config_panel_container, 
                                                     anchor="nw", tags="config_panel_container")
    
    # 实际内容的容器frame
    config_panel = ttk.Frame(config_panel_container, style="Card.TFrame", padding=15)
    config_panel.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # 右侧输出面板 - 添加标题和轻微阴影效果
    output_panel = ttk.Frame(app.training_page_frame, style="Card.TFrame", padding=15)
    output_panel.grid(row=0, column=1, sticky="nsew", padx=(0,5), pady=5)
    output_panel.rowconfigure(1, weight=1) # 让日志区域的行扩展
    output_panel.columnconfigure(0, weight=1) # 让日志区域的列扩展

    # === 配置面板内容 ===
    # 0. 系统环境
    env_section_content = create_section(config_panel, "🔧 " + i18n.get_text("system_environment_check", "系统环境检测"))

    env_buttons = ttk.Frame(env_section_content, style="Card.TFrame")
    env_buttons.pack(fill=tk.X, pady=(0, 10))

    ttk.Button(env_buttons, text=i18n.get_text("install_pytorch_cuda", "安装 PyTorch CUDA"),
               command=app.training_handler.install_pytorch_cuda,
               style="Success.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
    ttk.Button(env_buttons, text=i18n.get_text("fix_environment", "修复环境"),
               command=app.training_handler.fix_environment_issues,
               style="Warning.TButton").pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

    app.env_status_label = ttk.Label(env_section_content, 
                                     text=i18n.get_text("click_check_env", "点击按钮检测系统环境状态"),
                                     wraplength=350, style="Card.TLabel")
    app.env_status_label.pack(fill=tk.X, pady=(0, 15))

    # 1. 数据集配置
    dataset_section_content = create_section(config_panel, "📊 " + i18n.get_text("dataset_configuration", "数据集配置"))

    ttk.Button(dataset_section_content, text=i18n.get_text("select_dataset_yaml", "选择 datasets.yaml 文件"),
               command=app.training_handler.select_datasets_yaml,
               style="Primary.TButton").pack(fill=tk.X, pady=(0, 10))

    app.datasets_label = ttk.Label(dataset_section_content, textvariable=app.datasets_yaml_path,
                                   wraplength=350, style="Card.TLabel")
    app.datasets_label.pack(fill=tk.X, pady=(0, 15))
    app.datasets_yaml_path.set("📄 " + i18n.get_text("no_dataset_selected", "未选择数据集配置文件"))

    # 2. YOLO模型选择
    model_section_content = create_section(config_panel, "🤖 " + i18n.get_text("yolo_model_config", "YOLO 模型配置"))

    # 框架选择
    framework_frame = ttk.Frame(model_section_content, style="Card.TFrame")
    framework_frame.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(framework_frame, text=i18n.get_text("framework_version", "框架版本:"), style="Card.TLabel").pack(side=tk.LEFT)
    app.yolo_version_combo = ttk.Combobox(framework_frame, textvariable=app.selected_yolo_version_name,
                                          values=list(config.YOLO_VERSIONS.keys()),
                                          state="readonly", style="Modern.TCombobox", width=15)
    app.yolo_version_combo.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(10, 0))
    app.yolo_version_combo.bind('<<ComboboxSelected>>', app.training_handler.on_yolo_version_change)

    # 模型版本
    version_frame = ttk.Frame(model_section_content, style="Card.TFrame")
    version_frame.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(version_frame, text=i18n.get_text("model_version", "模型版本:"), style="Card.TLabel").pack(side=tk.LEFT)
    app.yolo_subversion_combo = ttk.Combobox(version_frame, textvariable=app.selected_yolo_subversion,
                                             state="disabled", style="Modern.TCombobox", width=15)
    app.yolo_subversion_combo.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(10, 0))
    app.yolo_subversion_combo.bind('<<ComboboxSelected>>', app.training_handler.on_yolo_subversion_change)

    ttk.Button(model_section_content, text=i18n.get_text("download_config_yolo", "下载/配置 YOLO 代码"),
               command=app.training_handler.setup_yolo_code,
               style="Modern.TButton").pack(fill=tk.X, pady=(0, 5))

    app.yolo_status_label = ttk.Label(model_section_content, textvariable=app.yolo_code_path,
                                      wraplength=350, style="Card.TLabel")
    app.yolo_status_label.pack(fill=tk.X, pady=(0, 15))
    app.yolo_code_path.set("⚙️ " + i18n.get_text("yolo_code_path_not_set", "未设置YOLO代码路径"))

    # 3. 预训练权重
    weights_section_content = create_section(config_panel, "⚖️ " + i18n.get_text("pretrained_weights", "预训练权重"))

    weights_select_frame = ttk.Frame(weights_section_content, style="Card.TFrame")
    weights_select_frame.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(weights_select_frame, text=i18n.get_text("weights_file", "权重文件:"), style="Card.TLabel").pack(side=tk.LEFT)
    app.weights_combo = ttk.Combobox(weights_select_frame, textvariable=app.weights_var,
                                     style="Modern.TCombobox", width=15)
    app.weights_combo.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(10, 0))

    weights_buttons = ttk.Frame(weights_section_content, style="Card.TFrame")
    weights_buttons.pack(fill=tk.X, pady=(0, 15))

    ttk.Button(weights_buttons, text="🌐 " + i18n.get_text("download_pretrained", "下载预训练模型"),
               command=app.training_handler.download_pretrained_weights,
               style="Success.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
    ttk.Button(weights_buttons, text="📁 " + i18n.get_text("browse_local_files", "浏览本地文件"),
               command=app.training_handler.browse_weights_file,
               style="Modern.TButton").pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

    # 4. 训练参数 - 使用两列网格布局优化显示
    params_section_content = create_section(config_panel, "⚙️ " + i18n.get_text("training_parameters", "训练参数"))
    params_grid = ttk.Frame(params_section_content, style="Card.TFrame")
    params_grid.pack(fill=tk.X, pady=(0, 5))
    params_grid.columnconfigure(1, weight=1)
    params_grid.columnconfigure(3, weight=1)

    # 创建参数输入行的辅助函数 - 使用网格布局
    def add_param_to_grid(parent, row, col, label_text, string_var, default_val="", width=8):
        string_var.set(default_val)
        ttk.Label(parent, text=label_text, style="Card.TLabel").grid(row=row, column=col*2, sticky="w", padx=(5,0), pady=5)
        entry = ttk.Entry(parent, textvariable=string_var, style="Modern.TEntry", width=width)
        entry.grid(row=row, column=col*2+1, sticky="ew", padx=5, pady=5)
        return entry

    # 两列参数布局
    add_param_to_grid(params_grid, 0, 0, i18n.get_text("training_epochs", "训练轮数:"), app.epochs_var, "100")
    add_param_to_grid(params_grid, 0, 1, i18n.get_text("batch_size", "批次大小:"), app.batch_var, "16")
    add_param_to_grid(params_grid, 1, 0, i18n.get_text("image_size", "图像尺寸:"), app.imgsz_var, "640")
    add_param_to_grid(params_grid, 1, 1, i18n.get_text("workers", "工作线程:"), app.workers_var, "4")

    # 设备选择
    ttk.Label(params_grid, text=i18n.get_text("device", "计算设备:"), style="Card.TLabel").grid(row=2, column=0, sticky="w", padx=(5,0), pady=5)
    app.device_combo = ttk.Combobox(params_grid, textvariable=app.device_var,
                                    values=["auto", "cpu", "0", "1", "2", "3"],
                                    style="Modern.TCombobox", width=8)
    app.device_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
    app.device_var.set("auto")

    ttk.Label(params_grid, text=i18n.get_text("experiment_name", "实验名称:"), style="Card.TLabel").grid(row=2, column=2, sticky="w", padx=(5,0), pady=5)
    run_name_entry = ttk.Entry(params_grid, textvariable=app.run_name_var, style="Modern.TEntry", width=8)
    run_name_entry.grid(row=2, column=3, sticky="ew", padx=5, pady=5)
    app.run_name_var.set("exp")

    # 输出目录
    ttk.Label(params_grid, text=i18n.get_text("output_dir", "输出目录:"), style="Card.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", padx=(5,0), pady=5)
    proj_entry = ttk.Entry(params_grid, textvariable=app.project_dir_var, style="Modern.TEntry")
    proj_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
    ttk.Button(params_grid, text="📁", command=app.training_handler.browse_project_dir,
            style="Modern.TButton", width=3).grid(row=3, column=3, sticky="w", padx=5, pady=5)
    app.project_dir_var.set(config.DEFAULT_PROJECT_DIR)

    # 5. 训练控制
    control_section_content = create_section(config_panel, "🎮 " + i18n.get_text("training_control", "训练控制"))

    control_buttons = ttk.Frame(control_section_content, style="Card.TFrame")
    control_buttons.pack(fill=tk.X)
    control_buttons.columnconfigure(0, weight=1)
    control_buttons.columnconfigure(1, weight=1)

    app.start_train_btn = ttk.Button(control_buttons, text="▶️ " + i18n.get_text("start_training", "开始训练"),
                                    command=app.training_handler.start_training,
                                    style="Primary.TButton")
    app.start_train_btn.grid(row=0, column=0, sticky="ew", padx=(0,5), pady=5)

    app.stop_train_btn = ttk.Button(control_buttons, text="⏹️ " + i18n.get_text("stop_training", "停止训练"),
                                    command=app.training_handler.stop_training,
                                    style="Danger.TButton", state="disabled")
    app.stop_train_btn.grid(row=0, column=1, sticky="ew", padx=(5,0), pady=5)

    # 断点恢复训练按钮（仅专业版可用）
    app.resume_train_btn = ttk.Button(control_buttons, text="⏸️ " + i18n.get_text("resume_training", "断点恢复训练"), command=app.training_handler.resume_training, style="Success.TButton")
    app.resume_train_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=(0,5))
    if hasattr(app, 'is_pro_version') and not app.is_pro_version():
        app.resume_train_btn.config(state="disabled")

    # === 输出面板内容 ===
    ttk.Label(output_panel, text="📊 " + i18n.get_text("training_output_progress", "训练输出与进度"), style="CardTitle.TLabel").grid(row=0, column=0, sticky="ew", pady=(0,15))

    # 创建一个容器给 Text 和 Scrollbar, 让这个容器在 output_panel 中扩展
    app.output_text_train_container = ttk.Frame(output_panel, style="Card.TFrame")
    app.output_text_train_container.grid(row=1, column=0, sticky="nsew")
    app.output_text_train_container.rowconfigure(0, weight=1)
    app.output_text_train_container.columnconfigure(0, weight=1)

    # 训练日志文本框
    app.output_text_train = tk.Text(app.output_text_train_container, height=10, wrap=tk.WORD, state=tk.DISABLED,
                                    bg="#313244", fg="#cdd6f4",
                                    relief=tk.SOLID, borderwidth=0, font=("Microsoft YaHei UI", 9),
                                    selectbackground="#89b4fa", selectforeground="#1e1e2e")
    app.output_text_train.grid(row=0, column=0, sticky="nsew", padx=(0,5), pady=5)

    train_log_scrollbar = ttk.Scrollbar(app.output_text_train_container, command=app.output_text_train.yview)
    train_log_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
    app.output_text_train.config(yscrollcommand=train_log_scrollbar.set)

    # 进度条等其他输出组件
    progress_frame = ttk.Frame(output_panel, style="Card.TFrame")
    progress_frame.grid(row=2, column=0, sticky="ew", pady=(10,5))
    progress_frame.columnconfigure(0, weight=1)

    app.progress_label = ttk.Label(progress_frame, text="⏳ " + i18n.get_text("waiting_start_training", "等待开始训练..."),
                                   style="Card.TLabel", font=("Microsoft YaHei UI", 10))
    app.progress_label.grid(row=0, column=0, sticky="w", pady=(0,5))

    app.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate', style="Modern.Horizontal.TProgressbar")
    app.progress_bar.grid(row=1, column=0, sticky="ew")
    
    # 为配置面板的滚动功能添加绑定
    def _on_config_canvas_configure(event):
        config_canvas.configure(scrollregion=config_canvas.bbox("all"))
        config_canvas.itemconfig(config_canvas_window, width=event.width)
    
    config_panel_container.bind("<Configure>", _on_config_canvas_configure)
    
    # 添加鼠标滚轮绑定
    def _on_mousewheel(event):
        if platform.system() == "Windows":
            config_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif platform.system() == "Darwin":  # macOS
            config_canvas.yview_scroll(int(-1*event.delta), "units")
    
    # 为Linux添加滚轮事件
    def _on_mousewheel_linux(event):
        if event.num == 4:  # 向上滚动
            config_canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # 向下滚动
            config_canvas.yview_scroll(1, "units")
    
    # 绑定滚轮事件
    if platform.system() == "Linux":
        config_canvas.bind("<Button-4>", _on_mousewheel_linux)
        config_canvas.bind("<Button-5>", _on_mousewheel_linux)
    else:
        config_canvas.bind("<MouseWheel>", _on_mousewheel)

    # 确保导出格式选择初始化
    if app.export_handler.export_formats:
        app.export_selected_format.set(list(app.export_handler.export_formats.keys())[0])
        app.export_handler.on_export_format_selected()

def create_export_page_layout(app):
    """创建导出页面布局 (现代化UI)"""
    app.export_page_frame = ttk.Frame(app.page_container, style="TFrame")
    # app.export_page_frame 在 _switch_to_page 时 pack(fill=tk.BOTH, expand=True)

    # 配置 export_page_frame 的列权重
    app.export_page_frame.columnconfigure(0, weight=0)  # 左侧设置面板
    app.export_page_frame.columnconfigure(1, weight=1)  # 右侧输出面板扩展
    app.export_page_frame.rowconfigure(0, weight=1)     # 整行扩展

    # 左侧设置面板 - 使用带滚动条的Canvas实现内容超出时可滚动
    settings_panel_outer = ttk.Frame(app.export_page_frame, style="TFrame", width=430) 
    settings_panel_outer.grid(row=0, column=0, sticky="ns", padx=(0,15))
    settings_panel_outer.grid_propagate(False)  # 固定宽度
    settings_panel_outer.rowconfigure(0, weight=1)
    settings_panel_outer.columnconfigure(0, weight=1)

    # 创建Canvas和滚动条
    settings_canvas = tk.Canvas(settings_panel_outer, bg="#1e1e2e", 
                               highlightthickness=0, bd=0)
    settings_canvas.grid(row=0, column=0, sticky="nsew")
    
    settings_scrollbar = ttk.Scrollbar(settings_panel_outer, orient="vertical", 
                                     command=settings_canvas.yview)
    settings_scrollbar.grid(row=0, column=1, sticky="ns")
    
    settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
    
    # 内部frame放置实际内容
    settings_panel_container = ttk.Frame(settings_canvas, style="TFrame")
    settings_canvas_window = settings_canvas.create_window((0, 0), window=settings_panel_container, 
                                                      anchor="nw", tags="settings_panel_container")
    
    # 实际内容的容器frame - 使用区块分组
    settings_panel = ttk.Frame(settings_panel_container, style="Card.TFrame", padding=15)
    settings_panel.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # 右侧输出面板
    output_feedback_frame = ttk.Frame(app.export_page_frame, style="Card.TFrame", padding=15)
    output_feedback_frame.grid(row=0, column=1, sticky="nsew", padx=(0,5), pady=5)
    output_feedback_frame.rowconfigure(1, weight=1)  # 日志区域扩展
    output_feedback_frame.columnconfigure(0, weight=1)  # 日志区域横向扩展

    # 1. 输入模型选择区块
    model_section_content = create_section(settings_panel, "📦 " + i18n.get_text("model_selection", "模型选择"))
    
    input_model_frame = ttk.Frame(model_section_content, style="Card.TFrame")
    input_model_frame.pack(fill=tk.X, pady=(0,5))
    input_model_frame.columnconfigure(0, weight=1)
    
    input_path_entry = ttk.Entry(input_model_frame, textvariable=app.export_input_model_path, 
                                state="readonly", style="Modern.TEntry")
    input_path_entry.grid(row=0, column=0, sticky="ew", padx=(0,5), pady=5)
    
    select_model_btn = ttk.Button(input_model_frame, text=i18n.get_text("select_model", "选择模型"), 
                                 command=app.export_handler.select_input_model, 
                                 style="Primary.TButton")
    select_model_btn.grid(row=0, column=1, sticky="e", pady=5)

    # 2. 输出配置区块
    output_section_content = create_section(settings_panel, "📤 " + i18n.get_text("output_configuration", "输出配置"))
    
    # 输出目录
    dir_frame = ttk.Frame(output_section_content, style="Card.TFrame")
    dir_frame.pack(fill=tk.X, pady=(0,5))
    dir_frame.columnconfigure(0, weight=1)
    
    ttk.Label(dir_frame, text=i18n.get_text("output_folder", "输出文件夹:"), style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=5)
    
    dir_select_frame = ttk.Frame(dir_frame, style="Card.TFrame")
    dir_select_frame.grid(row=1, column=0, sticky="ew")
    dir_select_frame.columnconfigure(0, weight=1)
    
    output_dir_entry = ttk.Entry(dir_select_frame, textvariable=app.export_output_dir_path, 
                               state="readonly", style="Modern.TEntry")
    output_dir_entry.grid(row=0, column=0, sticky="ew", padx=(0,5))
    
    select_dir_btn = ttk.Button(dir_select_frame, text="📁", 
                              command=app.export_handler.select_output_directory, 
                              style="Modern.TButton", width=3)
    select_dir_btn.grid(row=0, column=1, sticky="e")

    # 输出文件名
    name_frame = ttk.Frame(output_section_content, style="Card.TFrame")
    name_frame.pack(fill=tk.X, pady=(5,0))
    name_frame.columnconfigure(0, weight=1)
    
    ttk.Label(name_frame, text=i18n.get_text("output_filename", "输出文件名:"), style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=5)
    output_filename_entry = ttk.Entry(name_frame, textvariable=app.export_output_filename, 
                                    style="Modern.TEntry")
    output_filename_entry.grid(row=1, column=0, sticky="ew")

    # 3. 导出格式与选项区块
    format_section_content = create_section(settings_panel, "🔄 " + i18n.get_text("export_format_options", "导出格式与选项"))
    
    # 转换路径
    format_frame = ttk.Frame(format_section_content, style="Card.TFrame")
    format_frame.pack(fill=tk.X, pady=(0,10))
    format_frame.columnconfigure(0, weight=1)
    
    ttk.Label(format_frame, text=i18n.get_text("conversion_path", "转换路径:"), style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=5)
    app.export_format_combo = ttk.Combobox(format_frame, textvariable=app.export_selected_format,
                                         values=list(app.export_handler.export_formats.keys()), 
                                         state="readonly", style="Modern.TCombobox")
    app.export_format_combo.grid(row=1, column=0, sticky="ew")
    app.export_format_combo.bind('<<ComboboxSelected>>', app.export_handler.on_export_format_selected)

    # 动态选项区域 (由 ExportHandler填充)
    if hasattr(app, 'export_dynamic_options_frame') and app.export_dynamic_options_frame is not None:
        try:
            app.export_dynamic_options_frame.destroy()
        except Exception:
            pass
    app.export_dynamic_options_frame = ttk.Frame(format_section_content, style="Card.TFrame")
    app.export_dynamic_options_frame.pack(fill=tk.X, pady=(0,0))
    app.export_dynamic_options_frame.columnconfigure(0, weight=1)

    # 4. 操作控制区块
    control_section_content = create_section(settings_panel, "🎮 " + i18n.get_text("operation_control", "操作控制"))
    
    app.start_export_btn = ttk.Button(control_section_content, text="▶️ " + i18n.get_text("start_export", "开始导出"), 
                                     command=app.export_handler.start_export, 
                                     style="Primary.TButton")
    app.start_export_btn.pack(fill=tk.X, pady=(0,0))

    # 右侧日志面板内容 - 添加标题
    ttk.Label(output_feedback_frame, text="📤 " + i18n.get_text("export_log_status", "导出日志与状态"), style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0,10))

    # 导出日志文本区域
    app.export_log_text_container = ttk.Frame(output_feedback_frame, style="Card.TFrame")
    app.export_log_text_container.grid(row=1, column=0, sticky="nsew")
    app.export_log_text_container.rowconfigure(0, weight=1)
    app.export_log_text_container.columnconfigure(0, weight=1)

    app.output_text_export = tk.Text(app.export_log_text_container, height=8, wrap=tk.WORD, state=tk.DISABLED,
                                   bg="#313244", fg="#cdd6f4",
                                   relief=tk.SOLID, borderwidth=0, font=("Microsoft YaHei UI", 9),
                                   selectbackground="#89b4fa", selectforeground="#1e1e2e")
    app.output_text_export.grid(row=0, column=0, sticky="nsew", padx=(0,5), pady=5)

    export_log_scrollbar = ttk.Scrollbar(app.export_log_text_container, command=app.output_text_export.yview)
    export_log_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
    app.output_text_export.config(yscrollcommand=export_log_scrollbar.set)

    # 进度信息
    progress_frame = ttk.Frame(output_feedback_frame, style="Card.TFrame")
    progress_frame.grid(row=2, column=0, sticky="ew", pady=(10,0))
    progress_frame.columnconfigure(0, weight=1)
    
    app.export_progress_label = ttk.Label(progress_frame, 
                                       text=i18n.get_text("select_model_start_export", "选择模型并开始导出"), 
                                       style="Card.TLabel", font=("Microsoft YaHei UI", 10))
    app.export_progress_label.grid(row=0, column=0, sticky="w", pady=(0,5))
    
    app.export_progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, 
                                            length=200, mode='determinate', 
                                            style="Modern.Horizontal.TProgressbar")
    app.export_progress_bar.grid(row=1, column=0, sticky="ew")
    
    # 为设置面板的滚动功能添加绑定
    def _on_settings_canvas_configure(event):
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        settings_canvas.itemconfig(settings_canvas_window, width=event.width)
    
    settings_panel_container.bind("<Configure>", _on_settings_canvas_configure)
    
    # 添加鼠标滚轮绑定
    def _on_mousewheel(event):
        if platform.system() == "Windows":
            settings_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif platform.system() == "Darwin":  # macOS
            settings_canvas.yview_scroll(int(-1*event.delta), "units")
    
    # 为Linux添加滚轮事件
    def _on_mousewheel_linux(event):
        if event.num == 4:  # 向上滚动
            settings_canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # 向下滚动
            settings_canvas.yview_scroll(1, "units")
    
    # 绑定滚轮事件
    if platform.system() == "Linux":
        settings_canvas.bind("<Button-4>", _on_mousewheel_linux)
        settings_canvas.bind("<Button-5>", _on_mousewheel_linux)
    else:
        settings_canvas.bind("<MouseWheel>", _on_mousewheel)

    # 确保导出格式选择初始化
    if app.export_handler.export_formats:
        app.export_selected_format.set(list(app.export_handler.export_formats.keys())[0])
        app.export_handler.on_export_format_selected()

def create_inference_page_layout(app):
    """创建推理页面布局 (现代化UI)"""
    app.inference_page_frame = ttk.Frame(app.page_container, style="TFrame")
    app.inference_page_frame.columnconfigure(0, weight=0)  # 左侧设置面板
    app.inference_page_frame.columnconfigure(1, weight=1)  # 右侧输出面板扩展
    app.inference_page_frame.rowconfigure(0, weight=1)     # 整行扩展

    # 左侧设置面板 - 使用带滚动条的Canvas实现内容超出时可滚动
    settings_panel_outer = ttk.Frame(app.inference_page_frame, style="TFrame", width=430) 
    settings_panel_outer.grid(row=0, column=0, sticky="ns", padx=(0,15))
    settings_panel_outer.grid_propagate(False)
    settings_panel_outer.rowconfigure(0, weight=1)
    settings_panel_outer.columnconfigure(0, weight=1)

    settings_canvas = tk.Canvas(settings_panel_outer, bg="#1e1e2e", highlightthickness=0, bd=0)
    settings_canvas.grid(row=0, column=0, sticky="nsew")
    settings_scrollbar = ttk.Scrollbar(settings_panel_outer, orient="vertical", command=settings_canvas.yview)
    settings_scrollbar.grid(row=0, column=1, sticky="ns")
    settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
    settings_panel_container = ttk.Frame(settings_canvas, style="TFrame")
    settings_canvas_window = settings_canvas.create_window((0, 0), window=settings_panel_container, anchor="nw", tags="settings_panel_container")
    settings_panel = ttk.Frame(settings_panel_container, style="Card.TFrame", padding=15)
    settings_panel.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # 右侧输出面板
    output_feedback_frame = ttk.Frame(app.inference_page_frame, style="Card.TFrame", padding=15)
    output_feedback_frame.grid(row=0, column=1, sticky="nsew", padx=(0,5), pady=5)
    output_feedback_frame.rowconfigure(1, weight=1)
    output_feedback_frame.columnconfigure(0, weight=1)

    # 1. 模型选择区块
    model_section_content = create_section(settings_panel, "📦 " + i18n.get_text("model_selection", "选择推理模型"))
    input_model_frame = ttk.Frame(model_section_content, style="Card.TFrame")
    input_model_frame.pack(fill=tk.X, pady=(0,5))
    input_model_frame.columnconfigure(0, weight=1)
    app.infer_model_path_var = tk.StringVar()
    input_path_entry = ttk.Entry(input_model_frame, textvariable=app.infer_model_path_var, state="readonly", style="Modern.TEntry")
    input_path_entry.grid(row=0, column=0, sticky="ew", padx=(0,5), pady=5)
    select_model_btn = ttk.Button(input_model_frame, text="选择模型", command=lambda: app.inference_handler.select_model(), style="Primary.TButton")
    select_model_btn.grid(row=0, column=1, sticky="e", pady=5)

    # 2. 输入文件选择区块
    input_section_content = create_section(settings_panel, "🖼️/🎬 " + i18n.get_text("input_file_selection", "选择图片/视频"))
    input_file_frame = ttk.Frame(input_section_content, style="Card.TFrame")
    input_file_frame.pack(fill=tk.X, pady=(0,5))
    input_file_frame.columnconfigure(0, weight=1)
    app.infer_input_file_var = tk.StringVar()
    input_file_entry = ttk.Entry(input_file_frame, textvariable=app.infer_input_file_var, state="readonly", style="Modern.TEntry")
    input_file_entry.grid(row=0, column=0, sticky="ew", padx=(0,5), pady=5)
    select_file_btn = ttk.Button(input_file_frame, text="选择文件", command=lambda: app.inference_handler.select_input_file(), style="Primary.TButton")
    select_file_btn.grid(row=0, column=1, sticky="e", pady=5)

    # 3. 推理参数区块
    param_section_content = create_section(settings_panel, "⚙️ " + i18n.get_text("inference_parameters", "推理参数"))
    param_frame = ttk.Frame(param_section_content, style="Card.TFrame")
    param_frame.pack(fill=tk.X, pady=(0,5))
    ttk.Label(param_frame, text="置信度阈值:", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=5)
    app.infer_conf_var = tk.StringVar(value="0.25")
    conf_entry = ttk.Entry(param_frame, textvariable=app.infer_conf_var, style="Modern.TEntry", width=8)
    conf_entry.grid(row=0, column=1, sticky="w", padx=(5,0))
    ttk.Label(param_frame, text="IoU阈值:", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=5)
    app.infer_iou_var = tk.StringVar(value="0.45")
    iou_entry = ttk.Entry(param_frame, textvariable=app.infer_iou_var, style="Modern.TEntry", width=8)
    iou_entry.grid(row=1, column=1, sticky="w", padx=(5,0))

    # 4. 操作控制区块
    control_section_content = create_section(settings_panel, "🎮 " + i18n.get_text("operation_control", "操作控制"))
    app.start_infer_btn = ttk.Button(control_section_content, text="▶️ " + i18n.get_text("start_inference", "开始推理"), command=lambda: app.inference_handler.start_inference(), style="Primary.TButton")
    app.start_infer_btn.pack(fill=tk.X, pady=(0,0))

    # 右侧结果展示
    ttk.Label(output_feedback_frame, text="📊 " + i18n.get_text("inference_result_log", "推理结果与日志"), style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0,10))
    # 推理结果展示Frame+Canvas
    app.infer_result_display_frame = ttk.Frame(output_feedback_frame, style="Card.TFrame")
    app.infer_result_display_frame.grid(row=1, column=0, sticky="nsew", pady=(0,10))
    app.infer_result_display_frame.rowconfigure(0, weight=1)
    app.infer_result_display_frame.columnconfigure(0, weight=1)
    app.infer_result_canvas = tk.Canvas(app.infer_result_display_frame, bg="#232634", highlightthickness=0)
    app.infer_result_canvas.grid(row=0, column=0, sticky="nsew")
    app.infer_result_image_label = app.infer_result_display_frame
    # 推理日志输出区
    app.infer_output_text = tk.Text(output_feedback_frame, height=8, wrap=tk.WORD, state=tk.DISABLED,
                                   bg="#313244", fg="#cdd6f4", relief=tk.SOLID, borderwidth=0, font=("Microsoft YaHei UI", 9),
                                   selectbackground="#89b4fa", selectforeground="#1e1e2e")
    app.infer_output_text.grid(row=2, column=0, sticky="nsew", padx=(0,5), pady=5)
    infer_log_scrollbar = ttk.Scrollbar(output_feedback_frame, command=app.infer_output_text.yview)
    infer_log_scrollbar.grid(row=2, column=1, sticky="ns", pady=5)
    app.infer_output_text.config(yscrollcommand=infer_log_scrollbar.set)
    # 进度信息
    progress_frame = ttk.Frame(output_feedback_frame, style="Card.TFrame")
    progress_frame.grid(row=3, column=0, sticky="ew", pady=(10,0))
    progress_frame.columnconfigure(0, weight=1)
    app.infer_progress_label = ttk.Label(progress_frame, text="选择模型和文件后开始推理", style="Card.TLabel", font=("Microsoft YaHei UI", 10))
    app.infer_progress_label.grid(row=0, column=0, sticky="w", pady=(0,5))
    app.infer_progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate', style="Modern.Horizontal.TProgressbar")
    app.infer_progress_bar.grid(row=1, column=0, sticky="ew")
    # 滚动区域绑定
    def _on_settings_canvas_configure(event):
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        settings_canvas.itemconfig(settings_canvas_window, width=event.width)
    settings_panel_container.bind("<Configure>", _on_settings_canvas_configure)
    def _on_mousewheel(event):
        if platform.system() == "Windows":
            settings_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif platform.system() == "Darwin":
            settings_canvas.yview_scroll(int(-1*event.delta), "units")
    def _on_mousewheel_linux(event):
        if event.num == 4:
            settings_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            settings_canvas.yview_scroll(1, "units")
    if platform.system() == "Linux":
        settings_canvas.bind("<Button-4>", _on_mousewheel_linux)
        settings_canvas.bind("<Button-5>", _on_mousewheel_linux)
    else:
        settings_canvas.bind("<MouseWheel>", _on_mousewheel)