import sys
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class VideoImageManager:
    def __init__(self, thumb_width=200,
                 thumb_height=100):

        self.__cameras: list[(str, cv2.VideoCapture, tk.PhotoImage)] = []

        self.__thumb_width: int = thumb_width
        self.__thumb_height: int = thumb_height

        self.__current_video: cv2.VideoCapture = ...
        self.__current_video_title: str = ''
        self.__paused: bool = True

        self.__current_frame: np.ndarray = ...
        self.__current_frame_PI: tk.PhotoImage = ...
        self.__current_selected_area: tk.PhotoImage | None = None

        self.__window_width: int = 0
        self.__window_height: int = 0


    def get_current_video(self) -> cv2.VideoCapture:
        return self.__current_video

    def get_current_video_title(self) -> str:
        return self.__current_video_title

    def get_current_selected_area(self) -> tk.PhotoImage:
        return self.__current_selected_area

    def get_thumb_dimension(self) -> tuple[int, int]:
        return self.__thumb_width, self.__thumb_height

    def is_paused(self) -> bool:
        return self.__paused

    def set_paused(self, paused: bool):
        self.__paused = paused

    def set_video_dimensions(self, width: int, height: int):
        self.__window_width = width
        self.__window_height = height

    def set_video_progress(self, percentage: float):
        next_frame = int(percentage * self.__current_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__current_video.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

    def read_video(self) -> tk.PhotoImage | None:
        try:
            ret, frame = self.__current_video.read()
            if ret:

                frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                           (self.__window_width, self.__window_height))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))

                self.__current_frame = frame_resized
                self.__current_frame_PI = photo

                return photo
            else:
                return None
        except (OSError, cv2.error):
            print("Error while processing video...")
            sys.exit()

    def change_main_video(self, video: cv2.VideoCapture, video_title: str) -> cv2.VideoCapture:
        self.__paused = True
        self.__current_video_title = video_title
        self.__current_video = video
        self.__current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return self.__current_video

    def selected_area(self, coordinates: tuple[int, int, int, int]) -> tk.PhotoImage:
        alpha = 0.2  # Contrast control
        darkened_img = cv2.convertScaleAbs(self.__current_frame, alpha=alpha)
        x0, x1, y0, y1 = coordinates
        darkened_img[y0:y1, x0:x1] = self.__current_frame[y0:y1, x0:x1]

        image = Image.fromarray(darkened_img)
        photo = ImageTk.PhotoImage(image=image)
        cropped_image = image.crop((x0, y0, x1, y1))

        self.__current_frame_PI = photo
        self.__current_selected_area = ImageTk.PhotoImage(cropped_image)

        return photo
