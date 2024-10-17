import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

import cv2
from PIL import ImageTk

from .custom_progressbar import CustomProgressBar


class VideoSection(tk.Frame):
    def __init__(self, master, video_manager, re_id_manager, **kw):
        super().__init__(master, **kw)
        self.video_manager = video_manager
        self.re_id_manager = re_id_manager
        self.master = master
        self.cnv_video: tk.Canvas = tk.Canvas(self, height=self.get_height(),
                                              width=self.get_width(),
                                              highlightthickness=0)
        self.video_manager.set_video_dimensions(self.get_width(), self.get_height())

        self.btn_identification: tk.Button = tk.Button(self, text="Identify",
                                                       bg="dark grey",
                                                       width=50,
                                                       relief="ridge",
                                                       command=self.show_results)

        self.custom_progress_bar = CustomProgressBar(self, bg='light grey', height=15)

        self.rowconfigure((0, 1), weight=0)
        self.columnconfigure(0, weight=1)
        self.cnv_video.grid(row=0, column=0, padx=55, pady=(30, 0), sticky="w")
        self.cnv_video.grid_propagate(False)
        self.btn_identification.grid(row=2, column=0, sticky="n")
        self.custom_progress_bar.grid(row=1, column=0, padx=55, pady=(0, 20), sticky="nwe")
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.cnv_video.bind("<Button-1>", self.__start_drag)
        self.cnv_video.bind("<B1-Motion>", self.__on_drag)

        self.area_selection_enabled = True

    def get_width(self) -> int:
        return self.winfo_screenmmwidth() * 2

    def get_height(self) -> int:
        return self.winfo_screenmmheight() * 2

    def __read_video(self):
        try:
            photo = self.video_manager.read_video()

        except (cv2.error, RuntimeError) as e:
            print(f"Error while processing video:\n{e}")
            self.winfo_toplevel().destroy()
            sys.exit()

        else:
            if photo:
                self.cnv_video.create_image((0, 0), anchor=tk.NW, image=photo)
                self.custom_progress_bar.update_progress_bar(
                    self.video_manager.get_current_video().get(cv2.CAP_PROP_POS_FRAMES))
            else:
                self.check_play_button()

            if not self.video_manager.is_paused():
                self.cnv_video.after(8, self.__read_video)

    def seek_video(self, event: tk.Event) -> None:
        video_percentage = event.x / self.custom_progress_bar.progress_bar.winfo_width()
        self.video_manager.set_video_progress(video_percentage)
        self.custom_progress_bar.update_progress_bar(
            self.video_manager.get_current_video().get(cv2.CAP_PROP_POS_FRAMES))

    def check_play_button(self):
        if self.custom_progress_bar.button_pressed():
            self.video_manager.set_paused(False)
            self.__read_video()
        else:
            self.video_manager.set_paused(True)

    def change_main_video(self, video, camera_title):
        self.video_manager.change_main_video(video, camera_title)
        self.custom_progress_bar.reset_progress_bar(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__read_video()

    def __start_drag(self, event: tk.Event) -> None:
        if self.cnv_video is not None and self.area_selection_enabled:
            self.start_x = self.cnv_video.canvasx(event.x)
            self.start_y = self.cnv_video.canvasy(event.y)

    def __on_drag(self, event: tk.Event) -> None:
        if self.cnv_video is not None and self.area_selection_enabled:
            x0 = int(min(self.start_x, self.cnv_video.canvasx(event.x)))
            x1 = int(max(self.start_x, self.cnv_video.canvasx(event.x)))
            y0 = int(min(self.start_y, self.cnv_video.canvasy(event.y)))
            y1 = int(max(self.start_y, self.cnv_video.canvasy(event.y)))

            self.cnv_video.delete("drag_rectangle")
            self.cnv_video.create_rectangle(x0, y0, x1, y1, outline="white",
                                            fill="", tags="drag_rectangle")

            photo = self.video_manager.selected_area((x0, x1, y0, y1))

            self.cnv_video.create_image((0, 0), anchor=tk.NW, image=photo)

    def show_results(self) -> None:
        if self.video_manager.get_current_selected_area() is not None:
            self.disable_selection()

            query = ImageTk.getimage(self.video_manager.get_current_selected_area()).convert('RGB')
            future = self.executor.submit(self.re_id_manager.do_inference, query)
            future.add_done_callback(self.master.change_main_view)

    def enable_selection(self):
        self.btn_identification.config(state="normal")
        self.area_selection_enabled = True

    def disable_selection(self):
        self.btn_identification.config(state="disabled")
        self.area_selection_enabled = False


class ProbeSection(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master

        self.lbl_probe_title: tk.Label = tk.Label(self, text="Your probe", font=("Arial", 13),
                                                  fg="black", bg="light grey")

        self.lbl_probe: tk.Label = tk.Label(self, highlightthickness=0, border=0)
        self.__btn_change_probe: tk.Button = tk.Button(self, text="Change probe",
                                                       bg="dark grey",
                                                       command=self.master.restore_main_view,
                                                       width=50,
                                                       relief="ridge")

        self.rowconfigure((0, 1), weight=0)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.lbl_probe_title.grid(row=0, column=0, sticky="ew", pady=10)
        self.lbl_probe.grid(row=1, column=0, sticky="n", padx=15)
        self.__btn_change_probe.grid(row=2, column=0, sticky="n", pady=20, padx=0)

    def set_probe_image(self, current_probe: tk.PhotoImage) -> None:
        self.lbl_probe.config(image=current_probe)
