import os
import re
import sys
import tkinter as tk
from itertools import zip_longest
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


def load_cameras(video_dir='GUI/video',
                 video_type='mp4',
                 thumbnail_dim=(200, 100),
                 day: str = 'both',
                 video_manager=None) -> list:
    cameras = []
    try:
        if video_manager is None:
            raise RuntimeError('Video manager not specified')

        for root, _, files in os.walk(video_dir):
            if day != 'both' and not root.endswith(day):
                continue
            for file in files:
                path = os.path.join(root, file)
                if file.endswith(video_type):
                    video = cv2.VideoCapture(path)
                    if video.isOpened():
                        photo = video_manager.read_video(thumb_shape=thumbnail_dim,
                                                         video=video)
                        title = re.sub(r'day\d+_|\.mp4', '', file)
                        cameras.append((title, video, photo))

        return cameras
    except OSError:
        print("Error while loading images and cameras...")
        free_cameras(cameras)
        sys.exit()


def free_cameras(cameras_list):
    if cameras_list:
        for _, video, thumbnail in cameras_list:
            if video.isOpened():
                video.release()
            del thumbnail
        cameras_list.clear()


class CameraSection(tk.Frame):
    def __init__(self, master, video_section, day, **kw):
        super().__init__(master, **kw)

        self.canvas_window_id = -1
        self.video_section = video_section
        self.day = day

        self.scrollbar_camera: tk.Scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, bd=20,
                                                           width=20, bg="light grey")

        self.frm_container: tk.Frame = tk.Frame(self, highlightbackground="gray", border=2,
                                                bg="gray")

        self.canvas_cameras: tk.Canvas = tk.Canvas(self.frm_container, bg="white", highlightthickness=0,
                                                   borderwidth=0)

        self.frm_final_cameras = CamerasFrame(self.canvas_cameras, self.day,
                                              self.video_section,
                                              bg="white")

        self.frm_final_results = ResultsFrame(self.canvas_cameras, bg="white")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.frm_container.rowconfigure(0, weight=1)
        self.frm_container.columnconfigure(0, weight=1)
        self.frm_container.columnconfigure(1, weight=0)

        self.frm_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas_cameras.grid(row=0, column=0, sticky="nsew")
        self.frm_final_cameras.grid(row=0, column=0, sticky="nsew")
        self.frm_final_results.grid(row=0, column=0, sticky="nsew")
        self.frm_final_results.grid_remove()
        self.scrollbar_camera.grid(row=0, column=1, rowspan=2, sticky="nse")

        self.canvas_cameras.create_window((0, 0), anchor=tk.NW, window=self.frm_final_cameras, tags="cameras")

        self.scrollbar_camera.config(command=self.canvas_cameras.yview)  # Connect scrollbar to canvas
        self.canvas_cameras.config(yscrollcommand=self.scrollbar_camera.set)  # Connect canvas to scrollbar

        self.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas_cameras.bind("<Configure>", self.canvas_configure)

        self.frm_final_cameras.update_idletasks()
        self.canvas_cameras.config(scrollregion=self.canvas_cameras.bbox("all"))

    def on_mousewheel(self, event: tk.Event) -> None:
        self.canvas_cameras.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def display_results_view(self, results):
        self.frm_final_results.create_results_frames(results)
        self.canvas_cameras.delete(self.canvas_window_id)
        self.canvas_window_id = self.canvas_cameras.create_window((0, 0), anchor=tk.NW,
                                                                  window=self.frm_final_results,
                                                                  tags="result")
        self.canvas_cameras.bind("<Configure>", self.canvas_configure)
        self.frm_final_results.update_idletasks()
        self.canvas_cameras.config(scrollregion=self.canvas_cameras.bbox("all"))
        self.canvas_cameras.yview_moveto(0)

    def display_camera_view(self):
        if self.frm_final_results.results_imgs:
            self.frm_final_results.results_imgs.clear()
        self.canvas_cameras.delete(self.canvas_window_id)
        self.canvas_window_id = self.canvas_cameras.create_window((0, 0), anchor=tk.NW,
                                                                  window=self.frm_final_cameras,
                                                                  tags="cameras")

        self.canvas_cameras.bind("<Configure>", self.canvas_configure)
        self.frm_final_cameras.update_idletasks()
        self.canvas_cameras.config(scrollregion=self.canvas_cameras.bbox("all"))
        self.canvas_cameras.yview_moveto(0)

    def canvas_configure(self, event: tk.Event) -> None:
        self.canvas_cameras.itemconfig(self.canvas_window_id, width=event.width)


class CamerasFrame(tk.Frame):
    def __init__(self, master: tk.Canvas, day, video_section, **kw):
        super().__init__(master, **kw)

        self.master: tk.Canvas = master
        self.day = day
        self.video_section = video_section
        self.video_manager = self.video_section.video_manager
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)

        self.cameras = load_cameras(day=self.day, video_manager=self.video_manager)
        self.create_cameras_frames()

    def __del__(self):
        free_cameras(self.cameras)

    def create_cameras_frames(self) -> None:
        pad_x, pad_y = 13, 20
        for i, (camera_title, video, photo) in enumerate(self.cameras):
            width, height = self.video_manager.get_thumb_dimension()
            video_canvas = tk.Canvas(self, height=height, width=width)
            if i == 0:
                self.video_section.change_main_video(video, camera_title)

            video_canvas.grid(row=i, column=0, padx=pad_x, pady=pad_y, sticky="nw")

            video_canvas.create_image((0, 0), anchor=tk.NW, image=photo)

            video_canvas.bind("<Button-1>", lambda _, v=video, t=camera_title:
            self.video_section.change_main_video(v, t))

            sep = ttk.Separator(self, orient="horizontal")
            sep.grid(row=i, column=0, columnspan=2, sticky="sew")

            lbl_camera_title = tk.Label(self, text=camera_title,
                                        font=("Arial", 13), bg="white")
            lbl_camera_title.grid(row=i, column=1)


class ResultsFrame(tk.Frame):
    def __init__(self, master: tk.Canvas, **kw):
        super().__init__(master, **kw)

        self.master: tk.Canvas = master
        self.results_imgs = []
        self.lbl_result_title: tk.Label = tk.Label(self, text="Matches",
                                                   font=("Arial", 13),
                                                   fg="black", bg="white")

        self.rowconfigure(0, weight=0)
        self.columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.lbl_result_title.grid(row=0, column=0, sticky="ew", pady=(10, 30), columnspan=5)

    def create_results_frames(self, results: tuple) -> None:

        children_list: list[tk.Widget] = self.winfo_children()
        try:
            for i, (_, timestamp, camid, trackid, img_path, child) in enumerate(
                    zip_longest(*results, children_list[1:])):
                if img_path is None:
                    child.destroy()
                    children_list.remove(child)
                else:
                    img = None
                    if isinstance(img_path, str):
                        if os.path.isfile(img_path):
                            img = ImageTk.PhotoImage(Image.open(img_path).resize((92, 192)))

                    else:
                        img = ImageTk.PhotoImage(Image.fromarray(img_path.astype('uint8')).resize((92, 192)))
                    self.results_imgs.append(img)
                    number = "N: " + str(i + 1)
                    id_ = "ID: " + trackid.astype(str)
                    camera = "CAM: SAT_Camera" + camid.astype(str)
                    time = "TIME: " + timestamp.astype(str)

                    if child is None:
                        self.__create_result_frame(self, img, i,
                                                   number, id_, camera, time)
                    else:
                        self.__update_result_frames(child, img, number, id_, camera, time)


        except OSError:
            print("Error while retrieving results...")

    @staticmethod
    def __create_result_frame(window_: tk.Widget, img: tk.PhotoImage, frm_number: int, *texts: str) -> None:
        frm_result = tk.Frame(window_, bg="white")

        cnv_result = tk.Canvas(frm_result, width=img.width(), height=img.height(), highlightthickness=0)
        cnv_result.create_image(0, 0, anchor=tk.NW, image=img)
        cnv_result.grid(row=0, column=0, sticky="n", padx=(15, 0))

        for i, text in enumerate(texts):
            lbl_ = ResultsFrame.__create_result_frame_label(master=frm_result, text=text)
            lbl_.grid(row=i + 1, column=0, sticky="ew", padx=(30, 0))

        frm_result.grid(row=(frm_number // 5) + 1, column=frm_number % 5, sticky="nsew", pady=(0, 20))
        window_.winfo_children().append(frm_result)

    @staticmethod
    def __create_result_frame_label(*, master: tk.Widget = None,
                                    fg: str = "black", bg: str = "white",
                                    font: tuple[str, int] = ("Arial", 8),
                                    text: str = "", **kwargs) -> tk.Label:

        return tk.Label(master, fg=fg, bg=bg, font=font, text=text, **kwargs)

    @staticmethod
    def __update_result_frames(child, img: tk.PhotoImage, *new_text: str, **kwargs) -> None:
        inner_frm_children_list = child.winfo_children()
        inner_frm_children_list[0].config(width=img.width(), height=img.height())
        inner_frm_children_list[0].create_image(0, 0, anchor=tk.NW, image=img)
        for i, text in enumerate(new_text):
            inner_frm_children_list[i + 1].config(text=text, **kwargs)
