# annotation_handler.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import os
import glob
import config.config as config # Assuming your config file is named config.py
import platform # For platform-specific scroll binding
import math # For distance calculation

# 导入安全模块
try:
    from security import protect_annotation_feature
except ImportError:
    # 如果安全模块不存在，使用空装饰器
    def protect_annotation_feature(feature_name):
        def decorator(func): return func
        return decorator

class AnnotationHandler:
    def __init__(self, app):
        self.app = app
        self.image_dir = ""
        self.image_paths = []
        self.current_image_index = -1
        self.current_image_pil = None
        self.current_image_tk = None
        self.base_img_width = 0
        self.base_img_height = 0
        self.scale = 1.0
        self.image_display_x = 0
        self.image_display_y = 0
        self.annotations = []
        
        self.preview_rect_id = None
        self.preview_start_x_canvas = None
        self.preview_start_y_canvas = None

        self.selected_annotation_index = -1
        self.current_handles = [] # Stores canvas IDs of current handles
        self.is_resizing = False
        self.resizing_handle_type = None # e.g., 'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'
        self.resize_original_bbox_img = None # Original bbox in image coordinates
        self.resize_start_mouse_canvas = None # (x, y) of mouse press on canvas for resizing

        self.is_panning = False
        self.pan_start_x_canvas = 0
        self.pan_start_y_canvas = 0
        self.pan_start_image_display_x = 0
        self.pan_start_image_display_y = 0
        
        self.undo_stack = [] 
        self.redo_stack = []

    def _is_on_bbox_border(self, canvas_x, canvas_y, c_bbox, tolerance):
        """检查一个点是否在画布坐标系下的bbox边框上 (在容差范围内)
        c_bbox: [cx1, cy1, cx2, cy2] 画布坐标
        """
        cx1, cy1, cx2, cy2 = c_bbox
        # 点在扩展的矩形内
        if not (cx1 - tolerance <= canvas_x <= cx2 + tolerance and \
                cy1 - tolerance <= canvas_y <= cy2 + tolerance):
            return False
        # 点不在内部的缩小矩形内 (意味着它在边框区域)
        if (cx1 + tolerance <= canvas_x <= cx2 - tolerance and \
            cy1 + tolerance <= canvas_y <= cy2 - tolerance):
            return False
        return True

    def _push_to_undo_stack(self):
        # Deep copy of annotations and current selection state
        current_state = {
            'annotations': [{'bbox': list(ann['bbox']), 'class_name': ann['class_name']} for ann in self.annotations],
            'selected_annotation_index': self.selected_annotation_index
        }
        # To prevent redundant states if nothing changed effectively
        if self.undo_stack and self.undo_stack[-1] == current_state:
            return
        self.undo_stack.append(current_state)
        self.redo_stack.clear() # Clear redo stack on new action
        if len(self.undo_stack) > config.MAX_UNDO_HISTORY:
            self.undo_stack.pop(0) # Limit history size

    def _apply_state(self, state):
        self.annotations = [{'bbox': list(ann['bbox']), 'class_name': ann['class_name']} for ann in state['annotations']]
        self.selected_annotation_index = state['selected_annotation_index']
        self.update_annotation_listbox()
        self.redraw_canvas()

    def _init_default_classes(self):
        # (No changes from your previous version)
        self.app.class_details = []
        for i, c_def in enumerate(config.DEFAULT_CLASS_DEFINITIONS):
            color = c_def.get('color', config.FALLBACK_COLORS[i % len(config.FALLBACK_COLORS)])
            self.app.class_details.append({'name': c_def['name'], 'color': color})
        if hasattr(self.app, 'class_combobox'):
            self.app.class_combobox.configure(values=[cd['name'] for cd in self.app.class_details],
                                              state="readonly" if self.app.class_details else "disabled")
            if self.app.class_details:
                self.app.class_var.set(self.app.class_details[0]['name'])

    def load_image_dir(self):
        # (No changes from your previous version)
        if self.app.current_page != "annotate": return
        dir_path = filedialog.askdirectory(title="选择图片文件夹", parent=self.app.master)
        if not dir_path: return
        self.image_dir = dir_path
        self.image_paths = []
        for ext in config.IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
        self.image_paths.sort()
        if not self.image_paths:
            messagebox.showwarning("无图片", "选择的文件夹中未找到支持的图片格式。", parent=self.app.master)
            return
        
        # 添加图片数量限制检查
        try:
            from security import limit_training_images
            # 如果图片数量超过限制，limit_training_images函数会返回False
            if not limit_training_images(len(self.image_paths), parent=self.app.master):
                # 免费版限制为100张图片
                self.image_paths = self.image_paths[:100]
                messagebox.showinfo("图片数量限制", 
                                   "免费版最多支持100张图片进行标注。\n已自动选择前100张图片。\n升级到专业版可解除此限制。", 
                                   parent=self.app.master)
        except ImportError:
            # 如果安全模块不存在，不进行限制
            pass
            
        self.current_image_index = 0
        self.load_current_image()

    def load_current_image(self):
        # (No changes from your previous version, but ensure selection is reset)
        if self.app.current_page != "annotate" or not (0 <= self.current_image_index < len(self.image_paths)):
            return
        image_path = self.image_paths[self.current_image_index]
        try:
            self.current_image_pil = Image.open(image_path)
            self.base_img_width, self.base_img_height = self.current_image_pil.size
            self.selected_annotation_index = -1 # Reset selection on new image
            self.clear_handles()
            self.app.current_image_label.config(text=f"{self.current_image_index + 1}/{len(self.image_paths)}: {os.path.basename(image_path)}") # Update label
            self.reset_view(new_image_loaded=True) # This calls redraw_canvas
            self.load_annotations_for_current_image() # This loads annotations and calls redraw_canvas
            self.undo_stack.clear() # Clear history for new image
            self.redo_stack.clear()
            # Push initial state for the new image (empty or loaded annotations)
            self._push_to_undo_stack() # Push initial state
        except Exception as e:
            messagebox.showerror("加载图片错误", f"无法加载图片: {e}", parent=self.app.master)
            self.current_image_pil = None
            self.redraw_canvas()

    def next_image(self):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.image_paths: return
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_current_image()

    def prev_image(self):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.image_paths: return
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

    def reset_view(self, new_image_loaded=False):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.current_image_pil: return
        self.scale = 1.0
        canvas_width = self.app.canvas.winfo_width()
        canvas_height = self.app.canvas.winfo_height()
        if canvas_width > 0 and canvas_height > 0 and self.base_img_width > 0 and self.base_img_height > 0:
            self.image_display_x = (canvas_width - self.base_img_width * self.scale) / 2
            self.image_display_y = (canvas_height - self.base_img_height * self.scale) / 2
        else:
            self.image_display_x = 0
            self.image_display_y = 0
        self.redraw_canvas()

    def redraw_canvas(self):
        if self.app.current_page != "annotate" or not hasattr(self.app, 'canvas'): return
        self.app.canvas.delete("all") # Clear everything including handles
        if self.current_image_pil:
            self.display_image_on_canvas()
            self.redraw_canvas_annotations() # This will also draw handles for selected
        else:
            try:
                canvas_width = self.app.canvas.winfo_width()
                canvas_height = self.app.canvas.winfo_height()
                if canvas_width > 0 and canvas_height > 0:
                     self.app.canvas.create_text(canvas_width/2, canvas_height/2,
                                            text="请打开图片文件夹", fill=config.UI_FOREGROUND_COLOR, font=("Arial", 16))
            except tk.TclError: 
                pass

    def display_image_on_canvas(self):
        # (No changes from your previous version)
        if not self.current_image_pil: return
        try:
            display_width = int(self.base_img_width * self.scale)
            display_height = int(self.base_img_height * self.scale)
            if display_width <= 0 or display_height <= 0: return
            resample_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            img_resized = self.current_image_pil.resize((display_width, display_height), resample_method)
            self.current_image_tk = ImageTk.PhotoImage(img_resized)
            self.app.canvas.create_image(self.image_display_x, self.image_display_y, anchor=tk.NW, image=self.current_image_tk, tags="image")
        except Exception as e:
            print(f"Error displaying image: {e}")
            if hasattr(self.app, 'canvas') and self.app.canvas.winfo_exists():
                self.app.canvas.create_text(self.app.canvas.winfo_width()/2, self.app.canvas.winfo_height()/2,
                                            text=f"图片显示错误:\n{e}", fill=config.UI_ERROR_COLOR)

    def redraw_canvas_annotations(self): # Modified to include drawing handles
        if not self.current_image_pil: return
        self.clear_handles() # Clear old handles before redrawing annotations

        for i, ann in enumerate(self.annotations):
            x1_orig, y1_orig, x2_orig, y2_orig = ann['bbox']
            class_name = ann['class_name']
            
            # Transform original image coordinates to canvas coordinates
            cx1 = self.image_display_x + x1_orig * self.scale
            cy1 = self.image_display_y + y1_orig * self.scale
            cx2 = self.image_display_x + x2_orig * self.scale
            cy2 = self.image_display_y + y2_orig * self.scale
            
            box_color = self.get_class_color(class_name)
            outline_width = 2
            if i == self.selected_annotation_index:
                box_color = config.SELECTED_ANNOTATION_COLOR
                outline_width = 3 # Make selected box thicker
                self.draw_handles_for_bbox(cx1, cy1, cx2, cy2) # Draw handles for this selected box

            self.app.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=box_color, width=outline_width, tags=(f"ann_{i}", "annotation_box"))
            
            text_x, text_y = cx1 + 3, cy1 + 3
            if text_y > cy2 - 12: text_y = cy1 + 3 
            if text_x > cx2 - 20: text_x = cx1 + 3
            self.app.canvas.create_text(text_x, text_y, anchor=tk.NW, text=class_name, fill=config.ANNOTATION_TEXT_COLOR, font=("Arial", 10, "bold"), tags=(f"ann_text_{i}", "annotation_text"))

    def get_handle_positions(self, cx1, cy1, cx2, cy2):
        """ Returns a dictionary of handle types and their center (x,y) coordinates on canvas """
        s = config.HANDLE_SIZE # Visual size, not click radius
        return {
            'nw': (cx1, cy1), 'n': ((cx1 + cx2) / 2, cy1), 'ne': (cx2, cy1),
            'w': (cx1, (cy1 + cy2) / 2), 'e': (cx2, (cy1 + cy2) / 2),
            'sw': (cx1, cy2), 's': ((cx1 + cx2) / 2, cy2), 'se': (cx2, cy2)
        }

    def draw_handles_for_bbox(self, cx1, cy1, cx2, cy2):
        self.clear_handles() # Ensure old ones are gone
        handle_positions = self.get_handle_positions(cx1, cy1, cx2, cy2)
        s = config.HANDLE_SIZE

        for handle_type, (hx, hy) in handle_positions.items():
            # Draw small squares as handles
            handle_id = self.app.canvas.create_rectangle(
                hx - s, hy - s, hx + s, hy + s,
                fill=config.HANDLE_COLOR, outline="black", width=1,
                tags=(config.HANDLE_TAG, f"handle_{handle_type}")
            )
            self.current_handles.append(handle_id)

    def clear_handles(self):
        for handle_id in self.current_handles:
            self.app.canvas.delete(handle_id)
        self.current_handles = []
        self.app.canvas.config(cursor="cross") # Reset cursor when handles are cleared

    def get_class_color(self, class_name):
        # (No changes from your previous version)
        for cd in self.app.class_details:
            if cd['name'] == class_name:
                return cd['color']
        return config.UI_FOREGROUND_COLOR 

    def save_annotations(self):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.current_image_pil: 
            if self.app.current_page == "annotate":
                messagebox.showwarning("无图片", "没有加载图片，无法保存标注。", parent=self.app.master)
            return
        if not self.image_paths or self.current_image_index < 0:
            messagebox.showwarning("无图片", "图片信息错误，无法保存标注。", parent=self.app.master)
            return
        if not self.app.class_details:
            messagebox.showwarning("无类别", "类别未定义，无法保存标注。", parent=self.app.master)
            return
        image_path = self.image_paths[self.current_image_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_dir = self.image_dir if self.image_dir else os.path.dirname(image_path)
        txt_path = os.path.join(label_dir, base_name + ".txt")
        class_to_id = {cd['name']: i for i, cd in enumerate(self.app.class_details)}
        yolo_data = []
        img_w, img_h = self.base_img_width, self.base_img_height
        if img_w == 0 or img_h == 0:
            messagebox.showerror("错误", "图片尺寸信息错误，无法保存。", parent=self.app.master)
            return
        for ann in self.annotations:
            x1, y1, x2, y2 = ann['bbox']
            class_name = ann['class_name']
            if class_name not in class_to_id:
                self.app._add_to_output_log_queue(f"警告: 类别 '{class_name}' 未在类别列表中找到，跳过此标注。\n", is_error=True)
                continue
            class_id = class_to_id[class_name]
            dw = 1.0 / img_w; dh = 1.0 / img_h
            center_x = (x1 + x2) / 2.0 * dw; center_y = (y1 + y2) / 2.0 * dh
            width = (x2 - x1) * dw; height = (y2 - y1) * dh
            yolo_data.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        try:
            with open(txt_path, "w") as f: f.write("\n".join(yolo_data))
            self.app.status_bar.config(text=f"标注已保存: {os.path.basename(txt_path)}")
            self.app._add_to_output_log_queue(f"标注已保存: {txt_path}\n")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存标注到 {txt_path}: {e}", parent=self.app.master)

    def load_annotations_for_current_image(self):
        # (No changes from your previous version, but ensure redraw_canvas is called)
        self.annotations = []
        if not self.current_image_pil or not self.image_paths or self.current_image_index < 0: return
        image_path = self.image_paths[self.current_image_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_dir = self.image_dir if self.image_dir else os.path.dirname(image_path)
        txt_path = os.path.join(label_dir, base_name + ".txt")
        if not os.path.exists(txt_path): 
            self.update_annotation_listbox()
            self.redraw_canvas() 
            return
        if not self.app.class_details: 
            self.app._add_to_output_log_queue("警告: 类别未定义，无法加载标注。\n", is_error=True)
            return
        img_w, img_h = self.base_img_width, self.base_img_height
        if img_w == 0 or img_h == 0: return
        try:
            with open(txt_path, "r") as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id_float, center_x, center_y, width, height = map(float, parts)
                            class_id = int(class_id_float)
                            if not (0 <= class_id < len(self.app.class_details)):
                                self.app._add_to_output_log_queue(f"警告: 文件 {os.path.basename(txt_path)} 行 {line_num+1}: 无效的类别ID {class_id}。\n", is_error=True)
                                continue
                            class_name = self.app.class_details[class_id]['name']
                            x = center_x * img_w; y = center_y * img_h
                            w = width * img_w; h = height * img_h
                            x1 = int(x - w / 2); y1 = int(y - h / 2)
                            x2 = int(x + w / 2); y2 = int(y + h / 2)
                            self.annotations.append({'bbox': [x1, y1, x2, y2],'class_name': class_name})
                        except ValueError:
                            self.app._add_to_output_log_queue(f"警告: 文件 {os.path.basename(txt_path)} 行 {line_num+1}: 格式错误 '{line.strip()}'。\n", is_error=True)
                    elif line.strip():
                        self.app._add_to_output_log_queue(f"警告: 文件 {os.path.basename(txt_path)} 行 {line_num+1}: 格式不正确，应为5个值 '{line.strip()}'。\n", is_error=True)
        except Exception as e:
            self.app._add_to_output_log_queue(f"加载标注失败从 {os.path.basename(txt_path)}: {e}\n", is_error=True)
        self.update_annotation_listbox()
        self.redraw_canvas()

    def update_annotation_listbox(self):
        # (No changes from your previous version)
        if not hasattr(self.app, 'annotation_listbox'): return
        self.app.annotation_listbox.delete(0, tk.END)
        for i, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann['bbox']
            self.app.annotation_listbox.insert(tk.END, f"{i+1}: {ann['class_name']} [{x1},{y1},{x2},{y2}]")
        if 0 <= self.selected_annotation_index < len(self.annotations):
            self.app.annotation_listbox.selection_set(self.selected_annotation_index)
            self.app.annotation_listbox.activate(self.selected_annotation_index)
            self.app.annotation_listbox.see(self.selected_annotation_index)
        # else: # Don't automatically change selected_annotation_index here
            # self.selected_annotation_index = -1 # This line was causing deselection

    def manage_classes(self):
        # (No changes from your previous version)
        if self.app.current_page != "annotate": return
        class_info = "\n".join([f"- {cd['name']} ({cd['color']})" for cd in self.app.class_details])
        messagebox.showinfo("类别管理 (简化版)", 
                            f"当前类别:\n{class_info}\n\n类别在 config.py 中定义 (DEFAULT_CLASS_DEFINITIONS)。\n修改后需重启应用。", 
                            parent=self.app.master)

    @protect_annotation_feature("undo_redo")
    def undo_action(self):
        if self.app.current_page != "annotate": return
        
        # 先检查是否有权限使用撤销功能
        try:
            from security import check_feature_access
            if not check_feature_access("undo_redo_allowed"):
                # 如果没有权限，显示提示并返回
                from tkinter import messagebox
                messagebox.showwarning("许可证限制", 
                                      "撤销/重做功能需要专业版许可证。\n请升级到专业版解锁此功能。", 
                                      parent=self.app.master)
                return
        except ImportError:
            pass  # 如果安全模块不存在，继续执行
        
        if not self.undo_stack:
            self.app.status_bar.config(text="没有可撤销的操作")
            return
        
        # Current state goes to redo stack
        current_state_for_redo = {
            'annotations': [{'bbox': list(ann['bbox']), 'class_name': ann['class_name']} for ann in self.annotations],
            'selected_annotation_index': self.selected_annotation_index
        }
        self.redo_stack.append(current_state_for_redo)

        state_to_restore = self.undo_stack.pop()
        self._apply_state(state_to_restore)
        self.app.status_bar.config(text="操作已撤销")

    @protect_annotation_feature("undo_redo")
    def redo_action(self):
        if self.app.current_page != "annotate": return
        
        # 先检查是否有权限使用重做功能
        try:
            from security import check_feature_access
            if not check_feature_access("undo_redo_allowed"):
                # 如果没有权限，显示提示并返回
                from tkinter import messagebox
                messagebox.showwarning("许可证限制", 
                                      "撤销/重做功能需要专业版许可证。\n请升级到专业版解锁此功能。", 
                                      parent=self.app.master)
                return
        except ImportError:
            pass  # 如果安全模块不存在，继续执行
        
        if not self.redo_stack:
            self.app.status_bar.config(text="没有可重做的操作")
            return

        # Current state goes to undo stack before applying redo
        current_state_for_undo = {
            'annotations': [{'bbox': list(ann['bbox']), 'class_name': ann['class_name']} for ann in self.annotations],
            'selected_annotation_index': self.selected_annotation_index
        }
        self.undo_stack.append(current_state_for_undo)
        
        state_to_restore = self.redo_stack.pop()
        self._apply_state(state_to_restore)
        self.app.status_bar.config(text="操作已重做")

    def delete_selected_annotation_action(self):
        if self.app.current_page != "annotate": return
        if 0 <= self.selected_annotation_index < len(self.annotations):
            self._push_to_undo_stack() # Push state before deleting
            del self.annotations[self.selected_annotation_index]
            # Adjust selection after deletion
            if not self.annotations: # List is empty
                self.selected_annotation_index = -1
            elif self.selected_annotation_index >= len(self.annotations): # Was last item
                self.selected_annotation_index = len(self.annotations) - 1
            # Otherwise, selected_annotation_index effectively points to the next item, or stays if it was not the last one.
            # Or, more simply, just try to keep a valid index or deselect.
            if self.annotations:
                 self.selected_annotation_index = max(0, min(self.selected_annotation_index, len(self.annotations) -1))
            else:
                 self.selected_annotation_index = -1
            
            self.update_annotation_listbox()
            self.redraw_canvas() # This will clear handles if nothing is selected.
            self.app.status_bar.config(text="标注已删除") # Added status update

    def on_listbox_select(self, event): # Modified
        if self.app.current_page != "annotate": return
        selection = self.app.annotation_listbox.curselection()
        if selection:
            new_selection_index = selection[0]
            if new_selection_index != self.selected_annotation_index:
                self.selected_annotation_index = new_selection_index
                self.redraw_canvas() # Redraw to highlight and draw handles

    def get_clicked_handle(self, canvas_x, canvas_y):
        if self.selected_annotation_index == -1 or not self.annotations:
            return None

        ann = self.annotations[self.selected_annotation_index]
        x1_orig, y1_orig, x2_orig, y2_orig = ann['bbox']
        cx1 = self.image_display_x + x1_orig * self.scale
        cy1 = self.image_display_y + y1_orig * self.scale
        cx2 = self.image_display_x + x2_orig * self.scale
        cy2 = self.image_display_y + y2_orig * self.scale

        handle_positions = self.get_handle_positions(cx1, cy1, cx2, cy2)
        for handle_type, (hx, hy) in handle_positions.items():
            dist_sq = (canvas_x - hx)**2 + (canvas_y - hy)**2
            if dist_sq <= config.HANDLE_CLICK_RADIUS_SQUARED:
                return handle_type
        return None

    def get_clicked_annotation_index(self, canvas_x, canvas_y):
        for i in range(len(self.annotations) -1, -1, -1): # Iterate backwards for Z-order
            ann = self.annotations[i]
            x1_orig, y1_orig, x2_orig, y2_orig = ann['bbox']
            cx1 = self.image_display_x + x1_orig * self.scale
            cy1 = self.image_display_y + y1_orig * self.scale
            cx2 = self.image_display_x + x2_orig * self.scale
            cy2 = self.image_display_y + y2_orig * self.scale

            # Add a small tolerance for clicking borders
            tol = config.BORDER_CLICK_TOLERANCE
            if (cx1 - tol <= canvas_x <= cx2 + tol and
                cy1 - tol <= canvas_y <= cy2 + tol):
                # More precise check for being "on" the border or inside
                # For selection, inside is fine.
                return i
        return -1 # No annotation clicked

    def on_mouse_press(self, event): # Heavily Modified
        if self.app.current_page != "annotate" or not self.current_image_pil: return
        
        canvas_x, canvas_y = event.x, event.y
        # self._push_to_undo_stack() # Moved: Push state only when a confirmed change is made (draw, resize, delete) or selection that shows handles

        # 1. Check if clicking on a resize handle of the currently selected annotation
        if self.selected_annotation_index != -1 and self.current_handles:
            clicked_handle_type = self.get_clicked_handle(canvas_x, canvas_y)
            if clicked_handle_type:
                self._push_to_undo_stack() # Push before resize starts
                self.is_resizing = True
                self.resizing_handle_type = clicked_handle_type
                self.resize_original_bbox_img = list(self.annotations[self.selected_annotation_index]['bbox'])
                self.resize_start_mouse_canvas = (canvas_x, canvas_y)
                self.app.status_bar.config(text=f"调整大小: {self.resizing_handle_type}")
                return # Don't proceed to other actions

        # 2. Check if clicking on an existing annotation's BORDER to select it and show handles
        clicked_on_border_idx = -1
        for i in range(len(self.annotations) - 1, -1, -1): # Iterate backwards
            ann = self.annotations[i]
            x1_orig, y1_orig, x2_orig, y2_orig = ann['bbox']
            cx1 = self.image_display_x + x1_orig * self.scale
            cy1 = self.image_display_y + y1_orig * self.scale
            cx2 = self.image_display_x + x2_orig * self.scale
            cy2 = self.image_display_y + y2_orig * self.scale
            if self._is_on_bbox_border(canvas_x, canvas_y, [cx1, cy1, cx2, cy2], config.BORDER_CLICK_TOLERANCE):
                clicked_on_border_idx = i
                break
        
        if clicked_on_border_idx != -1:
            if self.selected_annotation_index != clicked_on_border_idx:
                self._push_to_undo_stack() # Push if selection changes to show handles
                self.selected_annotation_index = clicked_on_border_idx
                self.update_annotation_listbox() 
                self.redraw_canvas() # This will draw handles because it's selected
            # Else, it was already selected and border clicked again (could be start of a move later, for now no-op)
            return # Click was on a border, select and show handles

        # 3. If clicked on empty space (or inside an annotation but not on its border),
        #    deselect current annotation (if any) and clear handles.
        previous_selection = self.selected_annotation_index
        if self.selected_annotation_index != -1:
            self.selected_annotation_index = -1
            # self.update_annotation_listbox() # listbox will be cleared by redraw if no selection
            self.redraw_canvas() # This will clear handles
            if previous_selection != -1: # If there was a selection that is now cleared
                 self._push_to_undo_stack() # Record deselection if it results in handle removal

        # 4. Start drawing a new annotation (if not clicking handles or borders)
        if not self.app.class_details:
            messagebox.showwarning("无类别", "请先定义类别才能标注。", parent=self.app.master)
            return
        current_class = self.app.class_var.get()
        if not current_class and self.app.class_details:
            current_class = self.app.class_details[0]['name']
            self.app.class_var.set(current_class)
        elif not current_class and not self.app.class_details:
             return

        self.preview_start_x_canvas = canvas_x
        self.preview_start_y_canvas = canvas_y
        preview_color = self.get_class_color(current_class) or "cyan"
        self.preview_rect_id = self.app.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y,
            outline=preview_color, width=1, dash=(3,3)
        )
        self.app.status_bar.config(text=f"开始绘制: {current_class}")

    def on_mouse_drag(self, event): # Modified
        if self.app.current_page != "annotate": return
        canvas_x, canvas_y = event.x, event.y

        if self.is_resizing and self.selected_annotation_index != -1:
            if self.resize_original_bbox_img is None or self.resize_start_mouse_canvas is None:
                self.is_resizing = False # Safety break
                return

            # Deltas in canvas coordinates
            dx_canvas = canvas_x - self.resize_start_mouse_canvas[0]
            dy_canvas = canvas_y - self.resize_start_mouse_canvas[1]

            # Convert deltas to image coordinates
            dx_img = dx_canvas / self.scale
            dy_img = dy_canvas / self.scale

            # Get a mutable copy of original bbox in image coordinates
            new_bbox_img = list(self.resize_original_bbox_img) 
            orig_x1, orig_y1, orig_x2, orig_y2 = self.resize_original_bbox_img

            # Apply changes based on handle type
            if 'n' in self.resizing_handle_type: new_bbox_img[1] = orig_y1 + dy_img
            if 's' in self.resizing_handle_type: new_bbox_img[3] = orig_y2 + dy_img
            if 'w' in self.resizing_handle_type: new_bbox_img[0] = orig_x1 + dx_img
            if 'e' in self.resizing_handle_type: new_bbox_img[2] = orig_x2 + dx_img
            
            # Ensure x1 < x2 and y1 < y2, and min size
            final_x1, final_y1, final_x2, final_y2 = new_bbox_img
            min_size_img = config.MIN_ANNOTATION_SIZE_ON_CANVAS / self.scale

            if final_x1 > final_x2 - min_size_img: # x1 crossed x2 or too small
                if self.resizing_handle_type in ['w', 'nw', 'sw']: final_x1 = final_x2 - min_size_img
                else: final_x2 = final_x1 + min_size_img # e.g. dragging 'e' handle
            if final_y1 > final_y2 - min_size_img: # y1 crossed y2 or too small
                if self.resizing_handle_type in ['n', 'nw', 'ne']: final_y1 = final_y2 - min_size_img
                else: final_y2 = final_y1 + min_size_img # e.g. dragging 's' handle
            
            # Update annotation directly (integer coordinates)
            self.annotations[self.selected_annotation_index]['bbox'] = [
                int(max(0, min(final_x1, self.base_img_width))),
                int(max(0, min(final_y1, self.base_img_height))),
                int(max(0, min(final_x2, self.base_img_width))),
                int(max(0, min(final_y2, self.base_img_height)))
            ]
            self.redraw_canvas() # Redraws box and handles
            # No need to update listbox during drag, only on release

        elif self.preview_rect_id: # Drawing new annotation
            self.app.canvas.coords(self.preview_rect_id, 
                                   self.preview_start_x_canvas, self.preview_start_y_canvas, 
                                   canvas_x, canvas_y)
        elif self.is_panning: # Panning logic
            self.on_pan_drag(event) # Call existing pan drag


    def on_mouse_release(self, event): # Modified
        if self.app.current_page != "annotate": return

        action_taken_for_undo = False 

        if self.is_resizing:
            self.is_resizing = False
            self.resizing_handle_type = None
            # self.resize_original_bbox_img is already a list, no need to list() it again
            # self.resize_start_mouse_canvas = None # Not strictly needed to clear here
            if self.selected_annotation_index != -1: 
                current_bbox = self.annotations[self.selected_annotation_index]['bbox']
                self.annotations[self.selected_annotation_index]['bbox'] = [
                    int(max(0, min(current_bbox[0], self.base_img_width))),
                    int(max(0, min(current_bbox[1], self.base_img_height))),
                    int(max(0, min(current_bbox[2], self.base_img_width))),
                    int(max(0, min(current_bbox[3], self.base_img_height)))
                ]
                self.update_annotation_listbox() 
                self.redraw_canvas() 
            self.app.status_bar.config(text="调整完成")
            self.app.canvas.config(cursor="cross") 
            action_taken_for_undo = True # Resizing is an undoable action
            # return # Now we will fall through to push undo if action_taken_for_undo

        elif self.preview_rect_id: 
            end_x_canvas, end_y_canvas = event.x, event.y
            self.app.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None
            
            x1_c = min(self.preview_start_x_canvas, end_x_canvas)
            y1_c = min(self.preview_start_y_canvas, end_y_canvas)
            x2_c = max(self.preview_start_x_canvas, end_x_canvas)
            y2_c = max(self.preview_start_y_canvas, end_y_canvas)

            if (x2_c - x1_c) < config.MIN_ANNOTATION_SIZE_ON_CANVAS or \
               (y2_c - y1_c) < config.MIN_ANNOTATION_SIZE_ON_CANVAS:
                self.app.status_bar.config(text="标注太小，已取消")
                self.redraw_canvas()
                # return # Fall through, no undo needed for cancelled action
            else:
                x1_img = (x1_c - self.image_display_x) / self.scale
                y1_img = (y1_c - self.image_display_y) / self.scale
                x2_img = (x2_c - self.image_display_x) / self.scale
                y2_img = (y2_c - self.image_display_y) / self.scale
                
                x1_img = max(0, min(x1_img, self.base_img_width))
                y1_img = max(0, min(y1_img, self.base_img_height))
                x2_img = max(0, min(x2_img, self.base_img_width))
                y2_img = max(0, min(y2_img, self.base_img_height))

                if (x2_img - x1_img) > (config.MIN_ANNOTATION_SIZE_ON_CANVAS / self.scale) and \
                   (y2_img - y1_img) > (config.MIN_ANNOTATION_SIZE_ON_CANVAS / self.scale) :
                    selected_class_name = self.app.class_var.get() 
                    self.annotations.append({
                        'bbox': [int(x1_img), int(y1_img), int(x2_img), int(y2_img)],
                        'class_name': selected_class_name
                    })
                    self.selected_annotation_index = len(self.annotations) - 1
                    self.update_annotation_listbox()
                    self.redraw_canvas() 
                    self.app.status_bar.config(text=f"已添加标注: {selected_class_name}")
                    action_taken_for_undo = True # New annotation is an undoable action
                else:
                    self.app.status_bar.config(text="标注太小或超出边界，已取消")
                    self.redraw_canvas()
        
        elif self.is_panning: 
            self.on_pan_end(event) 

        if action_taken_for_undo:
            self._push_to_undo_stack()

    def get_handle_cursor(self, handle_type):
        # Standard Tkinter cursors. Some might not be ideal for all corners.
        if handle_type in ['n', 's']: return "sb_v_double_arrow"
        if handle_type in ['w', 'e']: return "sb_h_double_arrow"
        if handle_type == 'nw': return "top_left_corner" # or "size_nw_se" on some systems
        if handle_type == 'ne': return "top_right_corner" # or "size_ne_sw"
        if handle_type == 'sw': return "bottom_left_corner" # or "size_ne_sw"
        if handle_type == 'se': return "bottom_right_corner" # or "size_nw_se"
        return "crosshair" # Default

    def on_mouse_move_canvas(self, event): # Modified for cursor change
        if self.app.current_page != "annotate" or not self.current_image_pil: return
        
        canvas_x, canvas_y = event.x, event.y

        # Cursor updates only if not actively drawing, resizing or panning
        if not self.is_resizing and not self.is_panning and not self.preview_rect_id:
            new_cursor = "cross" # Default for drawing
            if self.selected_annotation_index != -1:
                handle_type = self.get_clicked_handle(canvas_x, canvas_y) # Check if over a handle
                if handle_type:
                    new_cursor = self.get_handle_cursor(handle_type)
                # else: # Optional: check if over border for a "move" cursor
                #     clicked_idx = self.get_clicked_annotation_index(canvas_x, canvas_y)
                #     if clicked_idx == self.selected_annotation_index:
                #          # More precise check if on border rather than just inside
                #          # For now, we'll skip dedicated move cursor on border hover
                #          pass
            self.app.canvas.config(cursor=new_cursor)

        # Status bar update (coordinates)
        img_x = (canvas_x - self.image_display_x) / self.scale
        img_y = (canvas_y - self.image_display_y) / self.scale
        img_x_clamped = max(0, min(img_x, self.base_img_width))
        img_y_clamped = max(0, min(img_y, self.base_img_height))

        # Only update status bar text if not actively resizing to avoid flicker
        if not self.is_resizing:
            if 0 <= img_x <= self.base_img_width and 0 <= img_y <= self.base_img_height:
                self.app.status_bar.config(text=f"图像坐标: ({img_x_clamped:.0f}, {img_y_clamped:.0f})  缩放: {self.scale:.2f}x")
            else:
                self.app.status_bar.config(text=f"画布坐标: ({canvas_x}, {canvas_y})  缩放: {self.scale:.2f}x")


    def on_mouse_wheel_zoom(self, event):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.current_image_pil: return
        zoom_factor = 1.1 if event.delta > 0 else (1 / 1.1)
        new_scale = self.scale * zoom_factor
        new_scale = max(config.MIN_SCALE, min(new_scale, config.MAX_SCALE))
        if new_scale == self.scale: return
        mouse_x_canvas = event.x; mouse_y_canvas = event.y
        img_coord_x_before_zoom = (mouse_x_canvas - self.image_display_x) / self.scale
        img_coord_y_before_zoom = (mouse_y_canvas - self.image_display_y) / self.scale
        self.scale = new_scale
        self.image_display_x = mouse_x_canvas - (img_coord_x_before_zoom * self.scale)
        self.image_display_y = mouse_y_canvas - (img_coord_y_before_zoom * self.scale)
        self.redraw_canvas()
        self.on_mouse_move_canvas(event) # Update status bar and cursor

    def on_mouse_wheel_scroll_vertical(self, event):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.current_image_pil: return
        if not (event.state & 0x0004): 
            delta = -config.SCROLL_STEP_VERTICAL if event.delta < 0 else config.SCROLL_STEP_VERTICAL
            self.image_display_y += delta
            self.redraw_canvas()

    def on_mouse_wheel_scroll_vertical_linux(self, event):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.current_image_pil: return
        if not (event.state & 0x0004): 
            if event.num == 4: self.image_display_y += config.SCROLL_STEP_VERTICAL
            elif event.num == 5: self.image_display_y -= config.SCROLL_STEP_VERTICAL
            self.redraw_canvas()

    def on_pan_start(self, event):
        # (No changes from your previous version)
        if self.app.current_page != "annotate" or not self.current_image_pil: return
        # Do not pan if a resize operation might be starting or an annotation is selected
        # Allowing pan only if no annotation is selected or if it's a clear middle click not near a handle
        if self.selected_annotation_index != -1 and self.get_clicked_handle(event.x, event.y) is not None:
            return # Prioritize handle interaction
        
        self.is_panning = True
        self.pan_start_x_canvas = event.x
        self.pan_start_y_canvas = event.y
        self.pan_start_image_display_x = self.image_display_x
        self.pan_start_image_display_y = self.image_display_y
        self.app.canvas.config(cursor="fleur")

    def on_pan_drag(self, event):
        # (No changes from your previous version, but ensure not resizing)
        if self.app.current_page != "annotate" or not self.is_panning or self.is_resizing: return # Added is_resizing check
        dx = event.x - self.pan_start_x_canvas
        dy = event.y - self.pan_start_y_canvas
        self.image_display_x = self.pan_start_image_display_x + dx
        self.image_display_y = self.pan_start_image_display_y + dy
        self.redraw_canvas()

    def on_pan_end(self, event):
        # (No changes from your previous version)
        if self.app.current_page != "annotate": return 
        if not self.is_panning: return 
        self.is_panning = False
        # Cursor reset is handled by on_mouse_move_canvas or clear_handles
        self.on_mouse_move_canvas(event)