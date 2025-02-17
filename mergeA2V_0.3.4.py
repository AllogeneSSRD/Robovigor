
from tkinter import *
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
import os
import time
from functools import partial
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.tools import cvsecs
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video
from tqdm import tqdm
from proglog import ProgressBarLogger

class CustomProgressLogger(ProgressBarLogger):
    """自定义进度日志记录器"""
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.bar = None
        self.logged_bars = 'all'
        self.ignored_bars = None
        self.start_time = None
        self.duration = None
        self.raw_log = []  # 新增：存储原始日志
        self.state['bars'] = None
        self.min_time_interval = 0
        self.ignore_bars_under = 0

    def __call__(self, message):
        """可调用接口现在会打印原始消息"""
        # 打印到控制台（调试用）
        print("[RAW MESSAGE]", repr(message))  # 显示原始消息的转义字符
        
        # 存储原始日志
        self.raw_log.append(message)
        
        # 同时显示到UI（可选）
        self.log_callback(f"[RAW] {repr(message)}\n")  # 使用repr显示特殊字符
        # 显示到UI
        # self.log_callback(f"[INFO] {message.strip()}\n")

        """实现可调用接口"""
        # 过滤进度信息（moviepy的进度日志包含'chunk'关键字）
        if 'chunk' in message:
            self._handle_progress(message)
        else:
            self.log_callback(f"[INFO] {message}\n")

    # def __call__(self, **kw):

    #     items = sorted(kw.items(), key=lambda kv: not kv[0].endswith('total'))

    #     for key, value in items:
    #         if '__' in key:
    #             bar, attr = key.split('__')
    #             if self.bar_is_ignored(bar):
    #                 continue
    #             kw.pop(key)
    #             if bar not in self.bars:
    #                 self.bars[bar] = dict(title=bar, index=-1,
    #                                         total=None, message=None)
    #             old_value = self.bars[bar][attr]

    #             if self.bar_is_logged(bar):
    #                 new_bar = (attr == 'index') and (value < old_value)
    #                 if (attr == 'total') or (new_bar):
    #                     self.bars[bar]['indent'] = self.log_indent
    #                 else:
    #                     self.log_indent = self.bars[bar]['indent']
    #                 self.log("[%s] %s: %s" % (bar, attr, value))
    #                 self.log_indent += self.bar_indent
    #             self.bars[bar][attr] = value
    #             self.bars_callback(bar, attr, value, old_value)
    #     self.state.update(kw)
    #     self.callback(**kw)

    def _handle_progress(self, message):
        """新增：打印进度相关的原始消息"""
        # print("[PROGRESS RAW]", repr(message))  # 进度相关消息的特殊标记

        print("消息类型:", type(message))  # 确认是str还是bytes
        print("消息长度:", len(message))
        
        import re
        pattern = r"\|.*\| (\d+)/(\d+) (\d+)% \[elapsed: (.*)\]"
        match = re.search(pattern, message)
        
        if match:
            current, total, percent, elapsed = match.groups()
            print(f"解析结果: 当前{current} 总数{total} 进度{percent}% 耗时{elapsed}")
            # 更新进度到UI...
        else:
            print("无法解析的进度消息格式:", repr(message))

        """处理进度信息"""
        # 示例消息："Moviepy - Writing audio in output.mp4\n
        #           Moviepy: Done writing audio in output.mp4 !\n
        #           Moviepy - Writing video in output.mp4\n
        #           |----------| 1000/1000 100% [elapsed: 00:00:05]"
        
        # 提取进度数值
        if "|----------|" in message:
            parts = message.split()
            progress = parts[2].split('/')[0]  # 获取当前进度
            total = parts[2].split('/')[1]
            percent = parts[3].replace('%','')
            
            progress_str = f"Progress: {percent}% ({progress}/{total})"
            self.log_callback(progress_str + '\r')  # 使用\r保持单行更新

    def bars_callback(self, bar, attr, value, old_value=None):
        """处理进度条更新"""
        if attr == 'total':
            self.duration = value
            self.bar = tqdm(total=value, file=sys.stdout, ncols=70)
        elif attr == 'index':
            elapsed_time = time.time() - self.start_time
            self.bar.n = value
            self.bar.refresh()
            percent = min(100, 100 * value / self.duration)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining = (self.duration - value) * elapsed_time / max(value, 1e-6)
            remaining = time.strftime("%H:%M:%S", time.gmtime(remaining))
            line = f"Progress: {percent:.1f}% ({value}/{self.duration}) [{elapsed}<{remaining}, {self.bar.format_dict['rate']:.2f}it/s]"
            self.log_callback(line)
            
    def callback(self, **kwargs):
        """主回调方法"""
        if self.start_time is None:
            self.start_time = time.time()
        if 'index' in kwargs:
            self.bars_callback(None, 'index', kwargs['index'])
            
    def close(self):
        if self.bar:
            self.bar.close()

class VideoProcessor:
    def __init__(self, video_path, audio_path, output_path, log_callback=None):
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_path = output_path # or os.path.splitext(video_path)[0] + "_output.mp4"
        self.logger = CustomProgressLogger(log_callback) if log_callback else None

    def process_video(self):
        audio_extension = os.path.splitext(self.audio_path)[1]
        if audio_extension == '.mp4':
            self.mp4_to_mp4()
        elif audio_extension == '.m4s':
            self.m4s_to_mp4()
        else:
            raise ValueError("Unsupported audio file extension")

    def _create_progress_bar(self, total):
        """创建带回调的tqdm进度条"""
        with tqdm(total=total, ncols=70, file=sys.stdout) as pbar:
            last = [0]
            def update(progress):
                increment = int(progress * total) - last[0]
                pbar.update(increment)
                last[0] = int(progress * total)
                if self.progress_callback:
                    self.progress_callback(pbar.format_meter(**pbar.format_dict))
            return update

    def mp4_to_mp4(self):
        video_clip = VideoFileClip(self.video_path)
        audio_clip = VideoFileClip(self.audio_path).audio
        final_clip = video_clip.set_audio(audio_clip)
        ffmpeg_write_video(
            final_clip,
            self.output_path,
            video_clip.fps,
            codec='libx264',
            logger=self.logger  # 传入自定义logger
        )
        if self.logger:
            self.logger.close()

    def m4s_to_mp4(self):
        video_clip = VideoFileClip(self.video_path)
        audio_clip = AudioFileClip(self.audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        ffmpeg_write_video(
            final_clip,
            self.output_path,
            video_clip.fps,
            codec='libx264',
            logger=self.logger
        )
        if self.logger:
            self.logger.close()


class VideoApp:
    def __init__(self, master):
        self.master = master
        master.title("Video Processor")
        master.geometry("800x600")


        self.create_input_fields()

        self.create_log_area()

        # 自定义重定向
        # self.tqdm_redirect = TqdmRedirect(self.log_area)
        # sys.stdout = self.tqdm_redirect
        # sys.stderr = self.tqdm_redirect  # 同时重定向标准错误
        # 添加原始日志显示开关（可选）
        self.show_raw = BooleanVar(value=True)
        ttk.Checkbutton(
            self.master,
            text="显示原始日志",
            variable=self.show_raw
        ).pack(anchor=NW, padx=10)

    def create_input_fields(self):
        frame = ttk.Frame(self.master, padding=10)
        frame.pack(fill=X)

        # Video Path
        ttk.Label(frame, text="Video Path:").grid(row=0, column=0, sticky=W)
        self.video_entry = ttk.Entry(frame, width=50)
        self.video_entry.grid(row=0, column=1, sticky=EW)
        ttk.Button(frame, text="Browse", command=self.browse_video).grid(row=0, column=2)
        
        # Audio Path
        ttk.Label(frame, text="Audio Path:").grid(row=1, column=0, sticky=W)
        self.audio_entry = ttk.Entry(frame, width=50)
        self.audio_entry.grid(row=1, column=1, sticky=EW)
        ttk.Button(frame, text="Browse", command=self.browse_audio).grid(row=1, column=2)
        
        # Output Path
        ttk.Label(frame, text="Output Path:").grid(row=2, column=0, sticky=W)
        self.output_entry = ttk.Entry(frame, width=50)
        self.output_entry.grid(row=2, column=1, sticky=EW)
        ttk.Button(frame, text="Browse", command=self.browse_output).grid(row=2, column=2)
        
        # 开始按钮
        ttk.Button(frame, text="Start Processing", command=self.start_processing).grid(row=3, column=1, pady=10)
        
        frame.columnconfigure(1, weight=1)

    def create_log_area(self):
        frame = ttk.Frame(self.master, padding=10)
        frame.pack(fill=BOTH, expand=True)

        self.log_area = scrolledtext.ScrolledText(frame, wrap=WORD, state='disabled')
        self.log_area.pack(fill=BOTH, expand=True)

    def browse_video(self):
        path = filedialog.askopenfilename()
        if path:
            self.video_entry.delete(0, END)
            self.video_entry.insert(0, path)

    def browse_audio(self):
        path = filedialog.askopenfilename()
        if path:
            self.audio_entry.delete(0, END)
            self.audio_entry.insert(0, path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4")
        if path:
            self.output_entry.delete(0, END)
            self.output_entry.insert(0, path)
    
    def get_output_path(self):
        output_path = self.output_entry.get()
        if not output_path:
            video_path = self.video_entry.get()
            if video_path:
                return os.path.splitext(video_path)[0] + "_output.mp4"
        return output_path
    
    def start_processing(self):
        video_path = self.video_entry.get()
        audio_path = self.audio_entry.get()
        output_path = self.get_output_path()

        if not all([video_path, audio_path]):
            print("Error: Video and Audio paths are required!")
            return

        def process():
            try:
                processor = VideoProcessor(
                    video_path,
                    audio_path,
                    output_path,
                    log_callback=self.update_progress
                )
                processor.process_video()
                self.append_log("\nProcessing completed successfully!\n")
            except Exception as e:
                self.append_log(f"\nError occurred: {str(e)}\n")
            # finally:
            #     if processor.logger:
            #         processor.logger.close()

        threading.Thread(target=process, daemon=True).start()

    def append_log(self, text):
        """普通日志追加"""
        self.log_area.after(0, self._safe_append_log, text)

    def _safe_append_log(self, text):
        self.log_area.configure(state='normal')
        self.log_area.insert(END, text)
        self.log_area.see(END)
        self.log_area.configure(state='disabled')

    # def update_progress(self, progress_str):
    #     """更新进度显示"""
    #     self.log_area.after(0, self._safe_update_progress, progress_str + '\r')  # 使用\r保持单行更新
    def update_progress(self, text):
        """支持多类型日志的更新方法"""
        if self.show_raw.get():  # 如果开启原始日志显示
            self._safe_append_full_log(text)
        else:
            self.log_area.after(0, self._safe_update_progress, text)


    # def _safe_update_progress(self, text):
    #     self.log_area.configure(state='normal')
    #     # 删除最后一行
    #     end_index = self.log_area.index("end-1c")
    #     last_line_start = self.log_area.search(r'\n', end_index, backwards=True, stopindex="1.0")
    #     if last_line_start:
    #         self.log_area.delete(last_line_start + "+1c", "end")
    #     else:
    #         self.log_area.delete("1.0", "end")
    #     # 插入新进度
    #     self.log_area.insert(END, text)
    #     self.log_area.see(END)
    #     self.log_area.configure(state='disabled')

    def _safe_update_progress(self, text):
        self.log_area.configure(state='normal')

        # 处理进度条覆盖逻辑
        if '\r' in text:  # 进度条更新
            text = text.replace('\r', '')
            # 删除最后一行
            end_index = self.log_area.index("end-1c linestart")
            self.log_area.delete(end_index, "end")
        else:  # 普通日志
            text += '\n'

        self.log_area.insert(END, text)
        self.log_area.see(END)
        self.log_area.configure(state='disabled')

    def _safe_append_full_log(self, text):
        """新增：完整显示原始日志的方法"""
        self.log_area.after(0, self._real_append_full_log, text)

    def _real_append_full_log(self, text):
        self.log_area.configure(state='normal')
        self.log_area.insert(END, text)
        self.log_area.see(END)
        self.log_area.configure(state='disabled')

class TqdmRedirect:
    """自定义tqdm输出重定向类"""
    def __init__(self, log_widget):
        self.log_widget = log_widget
        self.last_line_length = 0

    def write(self, s):
        s = s.strip().replace('\r', '')
        if s:  # 过滤空行和回车符
            self._update_widget(s)

    def _update_widget(self, s):
        self.log_widget.configure(state='normal')
        # 删除前一个进度条的字符数
        self.log_widget.delete(f"end-{self.last_line_length}c", "end")

        self.log_widget.insert(END, s)
        self.last_line_length = len(s)
        self.log_widget.see(END)
        self.log_widget.configure(state='disabled')
        self.log_widget.update_idletasks()
    def flush(self):
        pass




if __name__ == "__main__":
    root = Tk()
    app = VideoApp(root)
    root.mainloop()