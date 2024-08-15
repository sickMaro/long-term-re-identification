import os
import sys
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from tkinter import ttk
from cameras_manager import VideoImageManager
from identification import ReIdentificationManager
import cv2
import numpy as np
from PIL import Image, ImageTk


class Window:
    __MAIN_TITLE: str = 'Person re-identification - SATcase6video'
    __RESULT_TITLE: str = 'Result - SATcase6video'
    __WIDTH: int = 1
    __HEIGHT: int = 1
    __SCROLLBAR_WIDTH: int = 20
    __DIRECTORY: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    def __init__(self, cfg) -> None:
        self.__cfg = cfg
        self.__model = None
        self.video_image_manager = VideoImageManager()
        self.re_id_manager = ReIdentificationManager(self.__cfg)
        self.__window: tk.Tk = tk.Tk()
        self.__configure_window(main=True)
        self.__results: list[tk.PhotoImage] = []

        self.video_image_manager.load_images_and_cameras(self.__cfg.DATASETS.DAY)

        self.play_button, self.pause_button, self.volume_button = (
            self.video_image_manager.get_buttons_images())
        self.cameras = self.video_image_manager.get_cameras()

        self.__video: cv2.VideoCapture = ...

        self.__frm_main: tk.Frame = tk.Frame(self.__window, bg="light grey")
        self.__frm_video: tk.Frame = tk.Frame(self.__frm_main, bg="light grey")
        self.__frm_cameras: tk.Frame = tk.Frame(self.__window, bg="light grey")

        # size for the resized frames of the video
        Window.__WIDTH, Window.__HEIGHT = (Window.__get_frm_video_width(self),
                                           Window.__get_frm_video_height(self))

        self.video_image_manager.set_video_dimensions(Window.__WIDTH, Window.__HEIGHT)

        self.__lbl_title: tk.Label = tk.Label(self.__window, fg="black", bg="light grey",
                                              text=Window.__MAIN_TITLE, height=2, relief="raised",
                                              font=("Arial", 25))

        self.__cnv_video: tk.Canvas = tk.Canvas(self.__frm_video, height=Window.__HEIGHT, width=Window.__WIDTH,
                                                highlightthickness=0)

        self.__scrollbar_camera: tk.Scrollbar = tk.Scrollbar(self.__frm_cameras, orient=tk.VERTICAL, bd=20,
                                                             width=Window.__SCROLLBAR_WIDTH, bg="light grey")

        self.__frm_video_bar: tk.Frame = tk.Frame(self.__frm_video, bg="gray", height=15)

        self.__btn_play_video: tk.Button = tk.Button(self.__frm_video_bar, image=str(self.play_button),
                                                     relief="flat", height=20, width=20,
                                                     command=self.__button_pressed)

        self.__btn_video_volume: tk.Button = tk.Button(self.__frm_video_bar, image=str(self.volume_button),
                                                       relief="flat", height=20, width=20)

        self.__frm_container: tk.Frame = tk.Frame(self.__frm_cameras, highlightbackground="gray", border=2,
                                                  bg="gray")

        self.__canvas_cameras: tk.Canvas = tk.Canvas(self.__frm_container, bg="white", highlightthickness=0,
                                                     borderwidth=0)

        self.__frm_final_cameras: tk.Frame = tk.Frame(self.__canvas_cameras, bg="white")
        self.__frm_final_results: tk.Frame = tk.Frame(self.__canvas_cameras, bg="white")

        self.__lbl_result_title: tk.Label = tk.Label(self.__frm_final_results, text="Matches",
                                                     font=("Arial", 13), fg="black", bg="white")

        self.__progress_bar: ttk.Progressbar = ttk.Progressbar(self.__frm_video_bar, orient='horizontal',
                                                               mode='determinate')

        self.__btn_identification: tk.Button = tk.Button(self.__frm_video, text="Identify", bg="dark grey",
                                                         width=50, relief="ridge", command=self.__show_results)

        self.__frm_probe: tk.Frame = tk.Frame(self.__frm_main, bg="light grey")
        self.__lbl_probe_title: tk.Label = tk.Label(self.__frm_probe, text="Your probe", font=("Arial", 13),
                                                    fg="black", bg="light grey")

        self.__lbl_probe: tk.Label = tk.Label(self.__frm_probe, highlightthickness=0, border=0)
        self.__btn_change_probe: tk.Button = tk.Button(self.__frm_probe, text="Change probe", bg="dark grey",
                                                       command=self.__get_back, width=50, relief="ridge")

        self.__configure_and_grid_widgets()

        self.__canvas_cameras.create_window((0, 0), anchor=tk.NW, window=self.__frm_final_cameras, tags="cameras")

        self.__bind_events()
        self.__create_cameras_frames()

        self.__start_x: int = -1
        self.__start_y: int = -1

        self.__lock = threading.Lock()
        self.__event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.re_id_manager.load_solider_model()
        self.re_id_manager.load_face_detection_model()

    def __del__(self):
        if self.__results:
            self.__results.clear()

    def __configure_window(self, main: bool = False, multiplier: float = 1.5) -> None:
        if main:
            self.__window.state('zoomed')
            self.__window.resizable(True, True)

        self.__window.rowconfigure(1, weight=1)
        self.__window.columnconfigure(0, minsize=self.__window.winfo_screenmmwidth(), weight=1)
        self.__window.columnconfigure(1, minsize=self.__window.winfo_screenmmwidth() * multiplier, weight=1)

    def __configure_and_grid_widgets(self) -> None:

        Window.__configure_widgets((self.__frm_main, ((0, 0), (1, 0))),
                                   (self.__frm_video, ((0, 0), (1, 0))),
                                   (self.__frm_cameras, ((1,), (1, 0))),
                                   (self.__frm_video_bar, ((), (0, 1, 0))),
                                   (self.__frm_container, ((1,), (1, 0))),
                                   (self.__frm_final_cameras, ((1,), (0, 1))),
                                   (self.__frm_final_results, ((0,), (1, 1, 1, 1, 1))),
                                   (self.__frm_probe, ((0, 0), (1, 0))))

        self.__frm_main.grid(row=1, column=0, sticky="new", padx=(15, 15), pady=35)
        self.__frm_video.grid(row=0, column=0, sticky="nsew")
        self.__frm_cameras.grid(row=1, column=1, sticky="nsew", padx=(0, 15), pady=35)
        self.__lbl_title.grid(row=0, column=0, columnspan=2, sticky="new", padx=2, pady=2)
        self.__cnv_video.grid(row=0, column=0, padx=55, pady=(30, 0), sticky="w")
        self.__cnv_video.grid_propagate(False)
        self.__btn_identification.grid(row=2, column=0, sticky="n")
        self.__frm_video_bar.grid(row=1, column=0, padx=55, pady=(0, 20), sticky="nwe")
        self.__btn_play_video.grid(row=0, column=0, sticky="nsew")
        self.__btn_video_volume.grid(row=0, column=2, sticky="e")
        self.__frm_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.__canvas_cameras.grid(row=0, column=0, sticky="nsew")
        self.__frm_final_cameras.grid(row=0, column=0, sticky="nsew")
        self.__frm_final_results.grid(row=0, column=0, sticky="nsew")
        self.__lbl_result_title.grid(row=0, column=0, sticky="ew", pady=(10, 30), columnspan=5)
        self.__frm_final_results.grid_remove()
        self.__progress_bar.grid(row=0, column=1, sticky="ew")
        self.__scrollbar_camera.grid(row=0, column=1, rowspan=2, sticky="nse")

        self.__frm_probe.grid(row=0, column=0, sticky="nsew")
        self.__lbl_probe_title.grid(row=0, column=0, sticky="ew", pady=10)
        self.__lbl_probe.grid(row=1, column=0, sticky="n", padx=15)
        self.__btn_change_probe.grid(row=2, column=0, sticky="n", pady=20, padx=0)
        self.__frm_probe.grid_remove()

    def __canvas_configure(self, event: tk.Event) -> None:
        self.__canvas_cameras.itemconfig("cameras", width=event.width)

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
            if self.__btn_play_video["image"] == str(self.play_button):
                self.__btn_play_video["image"] = self.pause_button
                self.video_image_manager.set_paused(False)
                self.__read_video()
            else:
                self.__btn_play_video["image"] = self.play_button
                self.video_image_manager.set_paused(True)

    def __read_video(self):
        photo = self.video_image_manager.read_video()
        if photo:
            self.__cnv_video.create_image((0, 0), anchor=tk.NW, image=photo)
            self.__progress()
        else:
            self.__button_pressed()

        if not self.video_image_manager.is_paused():
            self.__cnv_video.after(8, self.__read_video)

    def __progress(self) -> None:
        if self.__progress_bar['value'] < self.__progress_bar["length"]:
            self.__progress_bar['value'] = (self.__video.get(cv2.CAP_PROP_POS_FRAMES)
                                            / self.__progress_bar["length"]) * 100

    def __seek_video(self, event: tk.Event) -> None:
        video_percentage = event.x / self.__progress_bar.winfo_width()
        self.video_image_manager.set_video_progress(video_percentage)
        self.__progress()

    @staticmethod
    def __configure_widgets(*widgets: tuple[tk.Misc, tuple[tuple[int | None, ...], tuple[int | None, ...]]]) \
            -> None:
        for widget, (rows, columns) in widgets:
            if rows is not None:
                for i, weight in enumerate(rows):
                    widget.rowconfigure(i, weight=weight)

            if columns is not None:
                for i, weight in enumerate(columns):
                    widget.columnconfigure(i, weight=weight)

    def __create_cameras_frames(self) -> None:
        pad_x, pad_y = 13, 20
        for i, (camera_title, video, photo) in enumerate(self.cameras):
            width, height = self.video_image_manager.get_thumb_dimension()
            video_canvas = tk.Canvas(self.__frm_final_cameras, height=height, width=width)
            if i == 0:
                self.__update_main_video(video, camera_title)

            video_canvas.grid(row=i, column=0, padx=pad_x, pady=pad_y, sticky="nw")

            video_canvas.create_image((0, 0), anchor=tk.NW, image=photo)

            video_canvas.bind("<Button-1>", lambda _, v=video, t=camera_title:
            self.__update_main_video(v, t))

            sep = ttk.Separator(self.__frm_final_cameras, orient="horizontal")
            sep.grid(row=i, column=0, columnspan=2, sticky="sew")

            lbl_camera_title = tk.Label(self.__frm_final_cameras, text=camera_title, font=("Arial", 13), bg="white")
            lbl_camera_title.grid(row=i, column=1)

        self.__frm_final_cameras.update_idletasks()
        self.__canvas_cameras.config(scrollregion=self.__canvas_cameras.bbox("all"))

    def __create_results_frames(self, results: tuple) -> None:
        distmat, timestamps, camids, trackids, imgs_paths = results
        indixes = np.argsort(distmat, axis=1)
        new_t = []
        new_c = []
        new_tr = []
        new_i = []
        for i in indixes[0]:
            new_t.append(timestamps[i])
            new_c.append(camids[i])
            new_tr.append(trackids[i])
            new_i.append(imgs_paths[i])
        results = new_t, new_c, new_tr, new_i

        children_list: list[tk.Widget] = self.__frm_final_results.winfo_children()
        try:
            for i, (timestamp, camid, trackid, img_path, child) in enumerate(
                    zip_longest(*results, children_list[1:])):
                if img_path is None:
                    child.destroy()
                    children_list.remove(child)
                else:
                    path = img_path
                    if os.path.isfile(path):
                        img = ImageTk.PhotoImage(Image.open(path).resize((92, 192)))
                        self.__results.append(img)
                        number = "N: " + str(i + 1)
                        id_ = "ID: " + trackid.astype(str)
                        camera = "CAM: SAT_Camera" + camid.astype(str)
                        time = "TIME: " + timestamp.astype(str)

                        if child is None:
                            Window.__create_result_frame(self.__frm_final_results, img, i,
                                                         number, id_, camera, time)
                        else:
                            Window.__update_result_frames(child, img, number, id_, camera, time)

            self.__frm_final_results.update_idletasks()
            self.__canvas_cameras.config(scrollregion=self.__canvas_cameras.bbox("all"))

        except OSError:
            print("Error while retrieving results...")
            self.__on_closing()

    @staticmethod
    def __create_result_frame(window_: tk.Widget, img: tk.PhotoImage, frm_number: int, *texts: str) -> None:
        frm_result = tk.Frame(window_, bg="white")

        cnv_result = tk.Canvas(frm_result, width=img.width(), height=img.height(), highlightthickness=0)
        cnv_result.create_image(0, 0, anchor=tk.NW, image=img)
        cnv_result.grid(row=0, column=0, sticky="n", padx=(30, 0))

        for i, text in enumerate(texts):
            lbl_ = Window.__create_result_frame_label(master=frm_result, text=text)
            lbl_.grid(row=i + 1, column=0, sticky="ew", padx=(30, 0))

        frm_result.grid(row=(frm_number // 5) + 1, column=frm_number % 5, sticky="nsew", pady=(0, 20))
        window_.winfo_children().append(frm_result)

    @staticmethod
    def __create_result_frame_label(*, master: tk.Widget = None, fg: str = "black", bg: str = "white",
                                    font: tuple[str, int] = ("Arial", 8), text: str = "") -> tk.Label:
        return tk.Label(master, fg=fg, bg=bg, font=font, text=text)

    @staticmethod
    def __update_result_frames(child, img: tk.PhotoImage, *new_text: str) -> None:
        inner_frm_children_list = child.winfo_children()
        inner_frm_children_list[0].config(width=img.width(), height=img.height())
        inner_frm_children_list[0].create_image(0, 0, anchor=tk.NW, image=img)
        for i, text in enumerate(new_text):
            inner_frm_children_list[i + 1].config(text=text)

    def __update_main_video(self, video: cv2.VideoCapture, camera_title: str) -> None:
        self.__video = self.video_image_manager.change_main_video(video, camera_title)
        self.__progress_bar["value"] = 0
        self.__progress_bar["length"] = self.__video.get(cv2.CAP_PROP_FRAME_COUNT)

        self.__btn_play_video["image"] = self.play_button
        self.__read_video()

    def __start_drag(self, event: tk.Event) -> None:
        if self.__cnv_video is not None:
            self.__start_x = self.__cnv_video.canvasx(event.x)
            self.__start_y = self.__cnv_video.canvasy(event.y)

    def __on_drag(self, event: tk.Event) -> None:
        if self.__cnv_video is not None:

            x0 = int(min(self.__start_x, self.__cnv_video.canvasx(event.x)))
            x1 = int(max(self.__start_x, self.__cnv_video.canvasx(event.x)))
            y0 = int(min(self.__start_y, self.__cnv_video.canvasy(event.y)))
            y1 = int(max(self.__start_y, self.__cnv_video.canvasy(event.y)))

            self.__cnv_video.delete("drag_rectangle")
            self.__cnv_video.create_rectangle(x0, y0, x1, y1, outline="white", fill="", tags="drag_rectangle")

            photo = self.video_image_manager.selected_area((x0, x1, y0, y1))

            self.__cnv_video.create_image((0, 0), anchor=tk.NW, image=photo)

    def __free_resources_on_closure(self) -> None:
        self.video_image_manager.__del__()

        '''directory = f'{self.__DIRECTORY}/datasets/my_dataset/query'
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)'''

        sys.exit()

    def __on_closing(self) -> None:
        self.__del__()
        self.__free_resources_on_closure()

    def show_window(self) -> None:
        self.__window.mainloop()

    def __show_results(self) -> None:
        if self.video_image_manager.get_current_selected_area() is not None:
            # self.__save_probe()
            self.__btn_identification.config(state="disabled")

            future = self.executor.submit(self.re_id_manager.inference_with_face_det_and_solider,
                                          ImageTk.getimage(self.video_image_manager.get_current_selected_area()).convert('RGB'))
            future.add_done_callback(self.__modify_view)

    def __save_probe(self) -> None:
        cropped_image = ImageTk.getimage(self.video_image_manager.get_current_selected_area())
        name = f'{self.video_image_manager.get_current_video_title()}_MILLIS_0_TRK-ID_0_TIMESTAMP_0-0-0.png'
        cropped_image.save(f'{self.__DIRECTORY}/datasets/my_dataset/query/{name}')

    def __modify_view(self, future):
        results = future.result()
        self.__lbl_title.config(text=Window.__RESULT_TITLE)
        self.__configure_window(multiplier=1.5)
        self.__frm_video.grid_remove()
        self.__frm_probe.grid()
        self.__lbl_probe.config(image=self.video_image_manager.get_current_selected_area())
        self.__canvas_cameras.delete("cameras")
        self.__canvas_cameras.create_window((0, 0), anchor=tk.NW, window=self.__frm_final_results, tags="result")
        self.__canvas_cameras.bind("<Configure>", self.__canvas_configure)
        self.__canvas_cameras.yview_moveto(0)
        self.__window.after(1000, self.__create_results_frames, results)

    def __get_back(self) -> None:
        self.__lbl_title.config(text=Window.__MAIN_TITLE)
        self.__configure_window()
        self.__frm_probe.grid_remove()
        self.__frm_video.grid()
        self.__canvas_cameras.delete("result")
        self.__canvas_cameras.create_window((0, 0), anchor=tk.NW, window=self.__frm_final_cameras, tags="cameras")
        self.__canvas_cameras.bind("<Configure>", self.__canvas_configure)
        self.__current_selected_area = None
        self.__btn_identification.config(state="normal")
        self.__frm_final_cameras.update_idletasks()
        self.__canvas_cameras.config(scrollregion=self.__canvas_cameras.bbox("all"))
        self.__canvas_cameras.yview_moveto(0)
        if self.__results:
            self.__results.clear()

        '''directory = f'{self.__DIRECTORY}/datasets/my_dataset/query'
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)'''
