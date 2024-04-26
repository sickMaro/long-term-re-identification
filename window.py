from tkinter import ttk
from PIL import Image, ImageTk
import os
import sys
import tkinter as tk
import cv2
import numpy as np
import re


class Window:
    __TITLE: str = 'Person re-identification - SATcase6video'
    __PLAY_BUTTON: tk.PhotoImage = None
    __PAUSE_BUTTON: tk.PhotoImage = None
    __VOLUME_BUTTON: tk.PhotoImage = None
    __WIDTH: int = 1
    __HEIGHT: int = 1
    __VIDEO_TYPE: str = ".mp4"
    __THUMB_WIDTH: int = 200
    __THUMB_HEIGHT: int = 100
    __SCROLLBAR_WIDTH: int = 20
    __CAMERAS: list[(str, cv2.VideoCapture, tk.PhotoImage)] = []
    __DIRECTORY: str = "C:/Users/robid/PycharmProjects/pythonProject2/video"

    def __init__(self) -> None:

        self.__window: tk.Tk = tk.Tk()
        self.__configure_window()

        Window.__load_images_and_cameras()

        self.__paused: bool = True
        self.__video: cv2.VideoCapture = ...
        self.__current_frame: np.ndarray | None = None
        self.__current_frame_PI: tk.PhotoImage = ...
        self.__current_selected_area: tk.PhotoImage | None = None

        self.__frm_video: tk.Frame = tk.Frame(self.__window, bg="light grey")
        self.__frm_cameras: tk.Frame = tk.Frame(self.__window, bg="light grey")

        # size for the resized frames of the video
        Window.__WIDTH, Window.__HEIGHT = (Window.__get_frm_video_width(self),
                                           Window.__get_frm_video_height(self))

        self.__lbl_title: tk.Label = tk.Label(self.__window, fg="black", bg="light grey", text=Window.__TITLE,
                                              height=2, relief="raised", font=("Arial", 25))

        self.__cnv_video: tk.Canvas = tk.Canvas(self.__frm_video, height=Window.__HEIGHT, width=Window.__WIDTH)

        self.__scrollbar_camera: tk.Scrollbar = tk.Scrollbar(self.__frm_cameras, orient=tk.VERTICAL, bd=20,
                                                             width=Window.__SCROLLBAR_WIDTH, bg="light grey")

        self.__frm_video_bar: tk.Frame = tk.Frame(self.__frm_video, bg="gray", height=15)

        self.__btn_play_video: tk.Button = tk.Button(self.__frm_video_bar, image=str(Window.__PLAY_BUTTON),
                                                     relief="flat", height=20, width=20,
                                                     command=self.__button_pressed)

        self.__btn_video_volume: tk.Button = tk.Button(self.__frm_video_bar, image=str(Window.__VOLUME_BUTTON),
                                                       relief="flat", height=20, width=20)

        self.__frm_container: tk.Frame = tk.Frame(self.__frm_cameras, highlightbackground="gray", border=2,
                                                  bg="gray")

        self.__canvas_cameras: tk.Canvas = tk.Canvas(self.__frm_container, bg="white", highlightthickness=0,
                                                     borderwidth=0)

        self.__frm_final_cameras: tk.Frame = tk.Frame(self.__canvas_cameras, bg="white")

        self.__progress_bar: ttk.Progressbar = ttk.Progressbar(self.__frm_video_bar, orient='horizontal',
                                                               mode='determinate')

        self.__configure_and_grid_widgets()

        self.__canvas_cameras.create_window((0, 0), anchor=tk.NW, window=self.__frm_final_cameras, tags="cameras")

        self.__bind_events()
        self.__create_cameras_frames()

        self.__frm_final_cameras.update_idletasks()
        self.__canvas_cameras.config(scrollregion=self.__canvas_cameras.bbox("all"))

        self.__start_x: int = -1
        self.__start_y: int = -1

    def __del__(self):
        self.__window.destroy()

    def __configure_window(self) -> None:
        self.__window.state('zoomed')
        self.__window.resizable(True, True)

        self.__window.rowconfigure(0, weight=0)
        self.__window.rowconfigure(1, weight=0)
        self.__window.columnconfigure(0, minsize=self.__window.winfo_screenmmwidth(), weight=1)
        self.__window.columnconfigure(1, minsize=self.__window.winfo_screenmmwidth() * 1.5, weight=1)

    def __configure_and_grid_widgets(self) -> None:

        Window.__configure_widgets(
            (self.__frm_video, ((0, 0), (1, 0))),
            (self.__frm_cameras, ((1,), (1, 0))),
            (self.__frm_video_bar, ((), (0, 1, 0))),
            (self.__frm_container, ((1,), (1, 0))),
            (self.__frm_final_cameras, ((), (0, 1))))

        self.__frm_video.grid(row=1, column=0, sticky="nsew", padx=(15, 15), pady=35)
        self.__frm_cameras.grid(row=1, column=1, sticky="nsew", padx=(0, 15), pady=35)
        self.__lbl_title.grid(row=0, column=0, columnspan=2, sticky="new")
        self.__cnv_video.grid(row=0, column=0, padx=55, pady=(30, 0), sticky="w")
        self.__cnv_video.grid_propagate(False)
        self.__frm_video_bar.grid(row=1, column=0, padx=55, pady=(0, 20), sticky="nwe")
        self.__btn_play_video.grid(row=0, column=0, sticky="nsew")
        self.__btn_video_volume.grid(row=0, column=2, sticky="e")
        self.__frm_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.__canvas_cameras.grid(row=0, column=0, sticky="nsew")
        self.__frm_final_cameras.grid(row=0, column=0, sticky="nsew")
        self.__progress_bar.grid(row=0, column=1, sticky="ew")
        self.__scrollbar_camera.grid(row=0, column=1, rowspan=2, sticky="nse")

    def __canvas_configure(self, event: tk.Event) -> None:
        self.__canvas_cameras.itemconfig("cameras", width=event.width)

    @classmethod
    def __load_images_and_cameras(cls) -> None:

        try:
            if not cls.__PLAY_BUTTON and not cls.__PAUSE_BUTTON and not cls.__VOLUME_BUTTON:
                cls.__PLAY_BUTTON = ImageTk.PhotoImage(Image.open("images/play-button.png").resize((25, 25)))
                cls.__PAUSE_BUTTON = ImageTk.PhotoImage(Image.open("images/pause-button.png").resize((25, 25)))
                cls.__VOLUME_BUTTON = ImageTk.PhotoImage(Image.open("images/volume.png").resize((30, 30)))

            if not cls.__CAMERAS:
                for file in os.listdir(cls.__DIRECTORY):
                    path = os.path.join(cls.__DIRECTORY, file)
                    if os.path.isfile(path) and file.endswith(cls.__VIDEO_TYPE):
                        video = cv2.VideoCapture(path)
                        if video.isOpened():
                            ret, frame = video.read()
                            if ret:
                                frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                                           (Window.__THUMB_WIDTH, Window.__THUMB_HEIGHT))
                                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                                cls.__CAMERAS.append((re.sub(r'day\d+_|\.mp4', '', file), video, photo))

        except OSError:
            print("Error while loading images and cameras...")
            cls.__free_resources_on_closure()

    def __bind_events(self) -> None:
        self.__cnv_video.bind("<Button-1>", self.__start_drag)
        self.__cnv_video.bind("<B1-Motion>", self.__on_drag)

        self.__scrollbar_camera.config(command=self.__canvas_cameras.yview)  # Connect scrollbar to canvas
        self.__canvas_cameras.config(yscrollcommand=self.__scrollbar_camera.set)  # Connect canvas to scrollbar

        self.__canvas_cameras.bind("<Configure>", self.__canvas_configure)
        self.__frm_cameras.bind_all("<MouseWheel>", self.__on_mousewheel)
        self.__progress_bar.bind("<Button-1>", self.__seek_video)

        self.__window.protocol("WM_DELETE_WINDOW", self.__on_closing)

    def __on_mousewheel(self, event: tk.Event) -> None:
        self.__canvas_cameras.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def __get_frm_video_width(self) -> int:
        return self.__frm_video.winfo_screenmmwidth() * 2

    def __get_frm_video_height(self) -> int:
        return self.__frm_video.winfo_screenmmheight() * 2

    def __button_pressed(self) -> None:
        if self.__btn_play_video["image"] is not None:
            if self.__btn_play_video["image"] == str(Window.__PLAY_BUTTON):
                self.__btn_play_video["image"] = Window.__PAUSE_BUTTON
                self.__paused = False
                self.__read_video()
            else:
                self.__btn_play_video["image"] = Window.__PLAY_BUTTON
                self.__paused = True

    def __read_video(self):
        try:
            ret, frame = self.__video.read()
            if ret:
                frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                           (Window.__WIDTH, Window.__HEIGHT))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                self.__cnv_video.create_image((0, 0), anchor=tk.NW, image=photo)
                self.__current_frame = frame_resized
                self.__current_frame_PI = photo
                self.__progress()
            else:
                self.__button_pressed()

            if not self.__paused:
                self.__cnv_video.after(9, self.__read_video)
        except (OSError, cv2.error):
            print("Error while processing video...")
            self.__on_closing()

    def __progress(self) -> None:
        if self.__progress_bar['value'] < self.__progress_bar["length"]:
            self.__progress_bar['value'] = (self.__video.get(cv2.CAP_PROP_POS_FRAMES)
                                            / self.__progress_bar["length"]) * 100

    def __seek_video(self, event: tk.Event) -> None:
        video_percentage = event.x / self.__progress_bar.winfo_width()
        self.__video.set(cv2.CAP_PROP_POS_FRAMES, int(video_percentage * self.__video.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.__progress()

    @staticmethod
    def __configure_widgets(*widgets: tuple[tk.Misc, tuple[tuple[int | None, ...], ...]]) -> None:
        for widget, (rows, columns) in widgets:
            if rows is not None:
                for i, weight in enumerate(rows):
                    widget.rowconfigure(i, weight=weight)

            if columns is not None:
                for i, weight in enumerate(columns):
                    widget.columnconfigure(i, weight=weight)

    def __create_cameras_frames(self) -> None:
        pad_x, pad_y = 13, 20
        for i, (camera_title, video, photo) in enumerate(Window.__CAMERAS):

            video_canvas = tk.Canvas(self.__frm_final_cameras,
                                     height=Window.__THUMB_HEIGHT, width=Window.__THUMB_WIDTH)
            if i == 0:
                video_canvas.grid(row=i, column=0, padx=pad_x, pady=pad_y + pad_y / 2, sticky="nw")
                self.__update_main_video(video)
            else:
                video_canvas.grid(row=i, column=0, padx=pad_x, pady=pad_y, sticky="nw")

            video_canvas.create_image((0, 0), anchor=tk.NW, image=photo)

            video_canvas.bind("<Button-1>", lambda _, v=video: self.__update_main_video(v))

            sep = ttk.Separator(self.__frm_final_cameras, orient="horizontal")
            sep.grid(row=i, column=0, columnspan=2, sticky="sew")

            lbl_camera_title = tk.Label(self.__frm_final_cameras, text=camera_title, font=("Arial", 13), bg="white")
            lbl_camera_title.grid(row=i, column=1)

    def __update_main_video(self, video: cv2.VideoCapture) -> None:
        self.__paused = True
        self.__video = video
        self.__video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.__progress_bar["value"] = 0
        self.__progress_bar["length"] = self.__video.get(cv2.CAP_PROP_FRAME_COUNT)

        self.__btn_play_video["image"] = Window.__PLAY_BUTTON
        self.__read_video()

    def __start_drag(self, event: tk.Event) -> None:
        if self.__cnv_video is not None:
            self.__start_x = self.__cnv_video.canvasx(event.x)
            self.__start_y = self.__cnv_video.canvasy(event.y)

    def __on_drag(self, event: tk.Event) -> None:
        if self.__cnv_video is not None:
            alpha = 0.2  # Contrast control

            x0 = int(min(self.__start_x, self.__cnv_video.canvasx(event.x)))
            x1 = int(max(self.__start_x, self.__cnv_video.canvasx(event.x)))
            y0 = int(min(self.__start_y, self.__cnv_video.canvasy(event.y)))
            y1 = int(max(self.__start_y, self.__cnv_video.canvasy(event.y)))

            self.__cnv_video.delete("drag_rectangle")
            self.__cnv_video.create_rectangle(x0, y0, x1, y1, outline="white", fill="", tags="drag_rectangle")

            darkened_img = cv2.convertScaleAbs(self.__current_frame, alpha=alpha)
            darkened_img[y0:y1, x0:x1] = self.__current_frame[y0:y1, x0:x1]

            photo = ImageTk.PhotoImage(image=Image.fromarray(darkened_img))

            self.__cnv_video.create_image((0, 0), anchor=tk.NW, image=photo)
            self.__current_selected_area = photo

    @classmethod
    def __free_resources_on_closure(cls) -> None:
        for image in (cls.__VOLUME_BUTTON, cls.__PAUSE_BUTTON, cls.__PLAY_BUTTON):
            if image:
                image.__del__()

        if cls.__CAMERAS:
            for _, video, thumbnail in cls.__CAMERAS:
                if video.isOpened():
                    video.release()
                thumbnail.__del__()
            cls.__CAMERAS.clear()

        sys.exit()

    def __on_closing(self):
        self.__del__()
        Window.__free_resources_on_closure()

    def show_window(self) -> None:
        self.__window.mainloop()


window = Window()
