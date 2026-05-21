import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import cv2
import numpy as np
import os
import math
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# 导入完美像素函数
from src.perfect_pixel import get_perfect_pixel

# ===================== 通用辅助函数 =====================
def cv2_imread_chinese(path):
    """读取中文路径的图像"""
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

def cv2_imwrite_chinese(path, img):
    """保存中文路径的图像（支持RGBA透明PNG）"""
    ext = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(ext, img)
    if result:
        encoded_img.tofile(path)

def cv2_to_photoimage(img, max_size=(450, 450)):
    """将cv2图像转换为tkinter的PhotoImage"""
    h, w = img.shape[:2]
    scale = min(max_size[0]/w, max_size[1]/h, 1)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if img_resized.shape[2] == 4:  # RGBA图像
        img_rgba = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(img_rgba)
    else:  # BGR图像
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil_img)

def create_checkerboard(size, cell_size=10):
    """生成棋盘格背景（用于透明预览）"""
    w, h = size
    checker = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            if (x//cell_size + y//cell_size) % 2 == 0:
                checker[y:y+cell_size, x:x+cell_size] = 200
            else:
                checker[y:y+cell_size, x:x+cell_size] = 150
    return cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)

def is_perfect_square(n):
    """验证是否为完全平方数"""
    if not isinstance(n, int) or n <= 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n

# ===================== 重构后的视频处理核心函数 =====================
def extract_frames_to_memory(video_path, total_frames=16):
    """仅在内存中提取视频帧，不保存到磁盘"""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    if len(all_frames) == 0:
        raise ValueError("视频无有效帧！")
    if len(all_frames) < total_frames:
        raise ValueError(f"视频总帧数({len(all_frames)})小于需要提取的帧数({total_frames})！")
    
    # 均匀提取指定帧数
    total_video_frames = len(all_frames)
    indices = np.linspace(0, total_video_frames - 1, total_frames, dtype=int)
    selected_frames = [all_frames[i] for i in indices]
    
    # 统一所有帧的尺寸（取最小宽高，居中裁剪）
    min_h = min(f.shape[0] for f in selected_frames)
    min_w = min(f.shape[1] for f in selected_frames)
    unified_frames = []
    for frame in selected_frames:
        h, w = frame.shape[:2]
        y_start = (h - min_h) // 2
        x_start = (w - min_w) // 2
        unified_frames.append(frame[y_start:y_start+min_h, x_start:x_start+min_w])
    
    return unified_frames, min_h, min_w

def create_raw_spritesheet(frames, grid_size):
    """在内存中合成原始精灵表，不保存到磁盘"""
    h, w = frames[0].shape[:2]
    spritesheet = np.zeros((grid_size * h, grid_size * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // grid_size
        col = idx % grid_size
        spritesheet[row*h : (row+1)*h, col*w : (col+1)*w] = frame
    return spritesheet

# ===================== 主程序 =====================
class PixelEditorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("像素风格化工具（视频+图像+抠图）")
        self.geometry("1000x650")
        
        # 菜单栏
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        menubar.add_command(label="视频处理", command=self.show_video_page)
        menubar.add_command(label="图像处理", command=self.show_image_page)
        menubar.add_command(label="抠图功能", command=self.show_matting_page)
        
        # 主容器
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # 初始化三个页面
        self.video_page = self.create_video_page()
        self.image_page = self.create_image_page()
        self.matting_page = self.create_matting_page()
        
        # 默认显示视频处理页面
        self.show_video_page()

    # -------------------- 页面切换函数 --------------------
    def show_video_page(self):
        self.image_page.pack_forget()
        self.matting_page.pack_forget()
        self.video_page.pack(fill=tk.BOTH, expand=True)
        self.title("像素风格化工具 - 视频处理")

    def show_image_page(self):
        self.video_page.pack_forget()
        self.matting_page.pack_forget()
        self.image_page.pack(fill=tk.BOTH, expand=True)
        self.title("像素风格化工具 - 图像处理")

    def show_matting_page(self):
        self.video_page.pack_forget()
        self.image_page.pack_forget()
        self.matting_page.pack(fill=tk.BOTH, expand=True)
        self.title("像素风格化工具 - 背景抠图")

    # -------------------- 视频处理页面（重构版） --------------------
    def create_video_page(self):
        page = ttk.Frame(self.main_container)
        
        # 1. 视频导入区域
        frame_input = ttk.LabelFrame(page, text="视频导入", padding=12)
        frame_input.pack(fill=tk.X, pady=(0, 10))
        
        self.video_path = None
        self.label_dnd = ttk.Label(frame_input, text="拖放视频文件到此处\n支持 MP4 / AVI / MOV / MKV", 
                                   relief=tk.GROOVE, padding=40, font=("微软雅黑", 12))
        self.label_dnd.pack(fill=tk.X, pady=5)
        self.label_dnd.drop_target_register(DND_FILES)
        self.label_dnd.dnd_bind('<<Drop>>', self.on_video_drop)
        
        # 控制区域：帧数输入 + 按钮横向排列
        frame_controls = ttk.Frame(frame_input)
        frame_controls.pack(pady=8)
        
        # 帧数输入框
        ttk.Label(frame_controls, text="提取帧数（平方数）：", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.frame_count_var = tk.StringVar(value="16")  # 默认16帧（4×4）
        self.entry_frame_count = ttk.Entry(frame_controls, textvariable=self.frame_count_var, width=8)
        self.entry_frame_count.pack(side=tk.LEFT, padx=(0, 10))
        
        # 按钮
        self.btn_select_video = ttk.Button(frame_controls, text="选择视频文件", command=self.select_video, width=15)
        self.btn_select_video.pack(side=tk.LEFT, padx=5)
        
        self.btn_process_video = ttk.Button(frame_controls, text="开始处理", command=self.process_video, 
                                            state=tk.DISABLED, width=15)
        self.btn_process_video.pack(side=tk.LEFT, padx=5)
        
        # 2. 处理日志区域
        frame_log = ttk.LabelFrame(page, text="处理日志", padding=12)
        frame_log.pack(fill=tk.BOTH, expand=True)
        
        self.text_log = tk.Text(frame_log, height=22, font=("Consolas", 11))
        self.text_log.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scroll_log = ttk.Scrollbar(frame_log, orient=tk.VERTICAL, command=self.text_log.yview)
        scroll_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_log.config(yscrollcommand=scroll_log.set)
        
        return page

    def log(self, msg):
        self.text_log.insert(tk.END, f"{msg}\n")
        self.text_log.see(tk.END)
        self.update()

    def on_video_drop(self, event):
        path = event.data.strip('{}')
        if os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.video_path = path
            self.label_dnd.config(text=f"已选择：{os.path.basename(path)}")
            self.btn_process_video.config(state=tk.NORMAL)
            self.log(f"✅ 已导入视频：{path}")

    def select_video(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if path:
            self.video_path = path
            self.label_dnd.config(text=f"已选择：{os.path.basename(path)}")
            self.btn_process_video.config(state=tk.NORMAL)
            self.log(f"✅ 已导入视频：{path}")

    def process_video(self):
        """重构后的处理流程：提取内存帧 → 合成原始精灵表 → 像素化整张精灵表"""
        if not self.video_path:
            messagebox.showwarning("警告", "请先导入视频！")
            return
        
        # 验证帧数输入
        try:
            total_frames = int(self.frame_count_var.get())
        except ValueError:
            messagebox.showerror("错误", "帧数必须为正整数！")
            return
        
        if not is_perfect_square(total_frames):
            messagebox.showerror("错误", "帧数必须为完全平方数！\n例如：9(3×3)、16(4×4)、25(5×5)、36(6×6)")
            return
        
        grid_size = int(math.isqrt(total_frames))
        self.btn_process_video.config(state=tk.DISABLED)
        try:
            self.text_log.delete(1.0, tk.END)
            
            # 步骤1：在内存中提取并统一帧尺寸
            self.log(f"🔄 步骤1/3：提取{total_frames}帧视频并统一尺寸...")
            frames, frame_h, frame_w = extract_frames_to_memory(self.video_path, total_frames=total_frames)
            self.log(f"✅ 帧提取完成，所有帧已统一为尺寸：{frame_h}×{frame_w}")
            
            # 步骤2：在内存中合成原始精灵表
            self.log(f"\n🔄 步骤2/3：合成{grid_size}×{grid_size}原始精灵表...")
            raw_spritesheet = create_raw_spritesheet(frames, grid_size)
            self.log(f"✅ 原始精灵表合成完成，尺寸：{raw_spritesheet.shape[0]}×{raw_spritesheet.shape[1]}")
            
            # 步骤3：对整张精灵表进行完美像素处理
            self.log("\n🔄 步骤3/3：对精灵表进行像素风格化处理...")
            # 转换为RGB格式（perfect_pixel要求）
            raw_spritesheet_rgb = cv2.cvtColor(raw_spritesheet, cv2.COLOR_BGR2RGB)
            w, h, processed_rgb = get_perfect_pixel(raw_spritesheet_rgb, sample_method="center", refine_intensity=0.3, debug=False)
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            self.log(f"✅ 像素化处理完成，最终精灵表尺寸：{w}×{h}")
            
            # 保存最终结果（仅保存处理后的精灵表和8倍放大版）
            output_path = "spritesheet_pixel.png"
            output_path_8x = "spritesheet_pixel_8x.png"
            cv2_imwrite_chinese(output_path, processed_bgr)
            # 8倍最近邻插值放大
            processed_8x = cv2.resize(processed_bgr, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
            cv2_imwrite_chinese(output_path_8x, processed_8x)
            
            self.log(f"\n🎉 所有处理完成！")
            self.log(f"✅ 最终像素化精灵表：{os.path.abspath(output_path)}")
            self.log(f"✅ 8倍放大精灵表：{os.path.abspath(output_path_8x)}")
            
            messagebox.showinfo("完成", f"视频处理完成！\n最终输出：\n1. 像素化精灵表：{output_path}\n2. 8倍放大版：{output_path_8x}")
        except Exception as e:
            self.log(f"\n❌ 处理失败：{str(e)}")
            messagebox.showerror("错误", f"处理失败：{str(e)}")
        finally:
            self.btn_process_video.config(state=tk.NORMAL)

    # -------------------- 图像处理页面（保持不变） --------------------
    def create_image_page(self):
        page = ttk.Frame(self.main_container)
        
        frame_main = ttk.Frame(page)
        frame_main.pack(fill=tk.BOTH, expand=True)
        
        frame_left = ttk.LabelFrame(frame_main, text="原始图像", padding=12)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.img_original = None
        self.img_original_path = None
        self.label_original = ttk.Label(frame_left, text="拖放图像到此处\n支持 PNG / JPG / BMP", 
                                        relief=tk.GROOVE, padding=50, font=("微软雅黑", 12))
        self.label_original.pack(fill=tk.BOTH, expand=True)
        self.label_original.drop_target_register(DND_FILES)
        self.label_original.dnd_bind('<<Drop>>', self.on_image_drop)
        
        btn_select_image = ttk.Button(frame_left, text="选择图像文件", command=self.select_image, width=18)
        btn_select_image.pack(pady=8)
        
        frame_right = ttk.LabelFrame(frame_main, text="处理后图像", padding=12)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.img_processed = None
        self.processed_size = (0, 0)
        self.label_processed = ttk.Label(frame_right, text="点击「处理」生成像素风格化图像\n（会自动弹出网格调试窗口）", 
                                         relief=tk.GROOVE, padding=50, font=("微软雅黑", 12))
        self.label_processed.pack(fill=tk.BOTH, expand=True)
        
        self.btn_process_image = ttk.Button(frame_right, text="处理", command=self.process_image, 
                                            state=tk.DISABLED, width=18)
        self.btn_process_image.pack(pady=(8, 4))
        
        self.btn_save_image = ttk.Button(frame_right, text="保存处理后图像", command=self.save_processed, 
                                         state=tk.DISABLED, width=18)
        self.btn_save_image.pack(pady=4)
        
        return page

    def on_image_drop(self, event):
        path = event.data.strip('{}')
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.load_original_image(path)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.load_original_image(path)

    def load_original_image(self, path):
        try:
            self.img_original_path = path
            self.img_original = cv2_imread_chinese(path)
            if self.img_original is None:
                raise ValueError("无法读取图像")
            photo = cv2_to_photoimage(self.img_original)
            self.label_original.config(image=photo, text="")
            self.label_original.photo = photo
            self.btn_process_image.config(state=tk.NORMAL)
            self.label_processed.config(image="", text="点击「处理」生成像素风格化图像\n（会自动弹出网格调试窗口）")
            self.img_processed = None
            self.processed_size = (0, 0)
            self.btn_save_image.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败：{str(e)}")

    def process_image(self):
        if self.img_original is None:
            messagebox.showwarning("警告", "请先导入原始图像！")
            return
        try:
            self.btn_process_image.config(state=tk.DISABLED)
            img_rgb = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
            w, h, out_rgb = get_perfect_pixel(img_rgb, sample_method="center", refine_intensity=0.3, debug=True)
            
            if w is None or h is None:
                raise ValueError("完美像素处理失败，无法检测网格")
            
            self.processed_size = (w, h)
            self.img_processed = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title("Input")
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.title(f"Pixel-perfect ({w}×{h})")
            plt.imshow(out_rgb)
            plt.axis("off")
            plt.show(block=False)
            
            photo = cv2_to_photoimage(self.img_processed)
            self.label_processed.config(image=photo, text=f"处理完成：{w}×{h}")
            self.label_processed.photo = photo
            self.btn_save_image.config(state=tk.NORMAL)
            
            messagebox.showinfo("完成", f"图像像素风格化处理完成！\n尺寸：{w}×{h}\n已弹出对比图和网格调试窗口")
        except Exception as e:
            messagebox.showerror("错误", f"处理失败：{str(e)}")
        finally:
            self.btn_process_image.config(state=tk.NORMAL)

    def save_processed(self):
        if self.img_processed is None:
            messagebox.showwarning("警告", "暂无处理后图像！")
            return
        path = filedialog.asksaveasfilename(
            title="保存处理后图像",
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPG图像", "*.jpg"), ("所有文件", "*.*")]
        )
        if path:
            try:
                cv2_imwrite_chinese(path, self.img_processed)
                w, h = self.processed_size
                out_8x = cv2.resize(self.img_processed, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
                path_8x = path.replace(".png", "_8x.png").replace(".jpg", "_8x.jpg")
                cv2_imwrite_chinese(path_8x, out_8x)
                
                messagebox.showinfo("完成", f"图像已保存！\n原始尺寸：{path}\n8倍放大：{path_8x}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败：{str(e)}")

    # -------------------- 抠图功能页面（保持不变） --------------------
    def create_matting_page(self):
        page = ttk.Frame(self.main_container)
        
        # 主容器（左右布局）
        frame_main = ttk.Frame(page)
        frame_main.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：原始图像（支持点击取色）
        frame_left = ttk.LabelFrame(frame_main, text="原始图像（点击背景取色）", padding=12)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.matting_original = None
        self.matting_original_hsv = None
        self.matting_scale = 1.0  # 图像显示缩放比例
        self.label_matting_original = ttk.Label(frame_left, text="拖放图像到此处\n支持 PNG / JPG / BMP", 
                                                relief=tk.GROOVE, padding=50, font=("微软雅黑", 12))
        self.label_matting_original.pack(fill=tk.BOTH, expand=True)
        self.label_matting_original.drop_target_register(DND_FILES)
        self.label_matting_original.dnd_bind('<<Drop>>', self.on_matting_drop)
        # 绑定鼠标点击取色事件
        self.label_matting_original.bind("<Button-1>", self.on_color_pick)
        
        btn_select_matting = ttk.Button(frame_left, text="选择图像文件", command=self.select_matting_image, width=18)
        btn_select_matting.pack(pady=8)
        
        # 右侧：抠图预览（棋盘格透明背景）
        frame_right = ttk.LabelFrame(frame_main, text="抠图预览（透明背景）", padding=12)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.matting_result = None
        self.label_matting_result = ttk.Label(frame_right, text="点击左侧图像背景取色\n或调整下方阈值", 
                                              relief=tk.GROOVE, padding=50, font=("微软雅黑", 12))
        self.label_matting_result.pack(fill=tk.BOTH, expand=True)
        
        # 阈值调整区域
        frame_threshold = ttk.LabelFrame(page, text="HSV阈值调整（拖动实时预览）", padding=12)
        frame_threshold.pack(fill=tk.X, pady=(10, 0))
        
        # H通道
        ttk.Label(frame_threshold, text="H（色调）：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.h_low = tk.IntVar(value=35)
        self.h_high = tk.IntVar(value=77)
        ttk.Scale(frame_threshold, from_=0, to=179, variable=self.h_low, command=self.update_matting_preview).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frame_threshold, text="-").grid(row=0, column=2, padx=2, pady=5)
        ttk.Scale(frame_threshold, from_=0, to=179, variable=self.h_high, command=self.update_matting_preview).grid(row=0, column=3, padx=5, pady=5)
        
        # S通道
        ttk.Label(frame_threshold, text="S（饱和度）：").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.s_low = tk.IntVar(value=43)
        self.s_high = tk.IntVar(value=255)
        ttk.Scale(frame_threshold, from_=0, to=255, variable=self.s_low, command=self.update_matting_preview).grid(row=0, column=5, padx=5, pady=5)
        ttk.Label(frame_threshold, text="-").grid(row=0, column=6, padx=2, pady=5)
        ttk.Scale(frame_threshold, from_=0, to=255, variable=self.s_high, command=self.update_matting_preview).grid(row=0, column=7, padx=5, pady=5)
        
        # V通道
        ttk.Label(frame_threshold, text="V（亮度）：").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.v_low = tk.IntVar(value=46)
        self.v_high = tk.IntVar(value=255)
        ttk.Scale(frame_threshold, from_=0, to=255, variable=self.v_low, command=self.update_matting_preview).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(frame_threshold, text="-").grid(row=1, column=2, padx=2, pady=5)
        ttk.Scale(frame_threshold, from_=0, to=255, variable=self.v_high, command=self.update_matting_preview).grid(row=1, column=3, padx=5, pady=5)
        
        # 操作按钮
        frame_matting_buttons = ttk.Frame(page)
        frame_matting_buttons.pack(pady=10)
        
        self.btn_reset_threshold = ttk.Button(frame_matting_buttons, text="重置阈值", command=self.reset_threshold, width=15)
        self.btn_reset_threshold.pack(side=tk.LEFT, padx=5)
        
        self.btn_save_matting = ttk.Button(frame_matting_buttons, text="保存透明PNG", command=self.save_matting_result, 
                                           state=tk.DISABLED, width=15)
        self.btn_save_matting.pack(side=tk.LEFT, padx=5)
        
        return page

    def on_matting_drop(self, event):
        path = event.data.strip('{}')
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.load_matting_image(path)

    def select_matting_image(self):
        path = filedialog.askopenfilename(
            title="选择需要抠图的图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.load_matting_image(path)

    def load_matting_image(self, path):
        try:
            self.matting_original = cv2_imread_chinese(path)
            if self.matting_original is None:
                raise ValueError("无法读取图像")
            self.matting_original_hsv = cv2.cvtColor(self.matting_original, cv2.COLOR_BGR2HSV)
            
            # 计算显示缩放比例
            h, w = self.matting_original.shape[:2]
            self.matting_scale = min(450/w, 450/h, 1)
            new_w, new_h = int(w*self.matting_scale), int(h*self.matting_scale)
            img_resized = cv2.resize(self.matting_original, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 显示原始图像
            photo = cv2_to_photoimage(img_resized)
            self.label_matting_original.config(image=photo, text="")
            self.label_matting_original.photo = photo
            
            # 重置预览和按钮
            self.label_matting_result.config(image="", text="点击左侧图像背景取色\n或调整下方阈值")
            self.matting_result = None
            self.btn_save_matting.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败：{str(e)}")

    def on_color_pick(self, event):
        """点击原始图像取色，自动设置HSV阈值"""
        if self.matting_original is None:
            return
        # 将点击坐标转换为原始图像坐标
        x = int(event.x / self.matting_scale)
        y = int(event.y / self.matting_scale)
        h, w = self.matting_original.shape[:2]
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)
        
        # 获取点击位置的HSV值
        h_val, s_val, v_val = self.matting_original_hsv[y, x]
        
        # 自动设置阈值范围（H±10，S和V从0到255）
        self.h_low.set(max(0, h_val - 10))
        self.h_high.set(min(179, h_val + 10))
        self.s_low.set(0)
        self.s_high.set(255)
        self.v_low.set(0)
        self.v_high.set(255)
        
        # 更新预览
        self.update_matting_preview()

    def update_matting_preview(self, *args):
        """根据当前HSV阈值更新抠图预览"""
        if self.matting_original is None:
            return
        try:
            # 获取当前阈值
            lower = np.array([self.h_low.get(), self.s_low.get(), self.v_low.get()])
            upper = np.array([self.h_high.get(), self.s_high.get(), self.v_high.get()])
            
            # 生成掩码
            mask = cv2.inRange(self.matting_original_hsv, lower, upper)
            mask_inv = cv2.bitwise_not(mask)  # 反转掩码：前景为白色
            
            # 生成带alpha通道的结果
            b, g, r = cv2.split(self.matting_original)
            self.matting_result = cv2.merge([b, g, r, mask_inv])
            
            # 生成棋盘格背景用于预览
            h, w = self.matting_original.shape[:2]
            checker = create_checkerboard((w, h))
            # 将前景叠加到棋盘格上
            fg = cv2.bitwise_and(self.matting_original, self.matting_original, mask=mask_inv)
            bg = cv2.bitwise_and(checker, checker, mask=mask)
            preview = cv2.add(fg, bg)
            
            # 缩放并显示预览
            preview_resized = cv2.resize(preview, (int(w*self.matting_scale), int(h*self.matting_scale)), 
                                        interpolation=cv2.INTER_NEAREST)
            photo = cv2_to_photoimage(preview_resized)
            self.label_matting_result.config(image=photo, text="")
            self.label_matting_result.photo = photo
            self.btn_save_matting.config(state=tk.NORMAL)
        except Exception as e:
            print(f"预览更新失败：{str(e)}")

    def reset_threshold(self):
        """重置为默认绿幕阈值"""
        self.h_low.set(35)
        self.h_high.set(77)
        self.s_low.set(43)
        self.s_high.set(255)
        self.v_low.set(46)
        self.v_high.set(255)
        if self.matting_original is not None:
            self.update_matting_preview()

    def save_matting_result(self):
        """保存带透明通道的PNG图像"""
        if self.matting_result is None:
            messagebox.showwarning("警告", "暂无抠图结果！")
            return
        path = filedialog.asksaveasfilename(
            title="保存透明PNG图像",
            defaultextension=".png",
            filetypes=[("PNG图像（支持透明）", "*.png"), ("所有文件", "*.*")]
        )
        if path:
            try:
                cv2_imwrite_chinese(path, self.matting_result)
                messagebox.showinfo("完成", f"透明PNG已保存至：{path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败：{str(e)}")

if __name__ == "__main__":
    app = PixelEditorApp()
    app.mainloop()
