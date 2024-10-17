import sys
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk, Image


def load_buttons_images(images_dir):
    play_button_img = pause_button_img = volume_button_img = None
    try:
        play_button_img = Image.open(f"{images_dir}/play-button.png").resize((25, 25))
        pause_button_img = Image.open(f"{images_dir}/pause-button.png").resize((25, 25))
        volume_button_img = Image.open(f"{images_dir}/volume.png").resize((30, 30))

        play_button = ImageTk.PhotoImage(play_button_img)
        pause_button = ImageTk.PhotoImage(pause_button_img)
        volume_button = ImageTk.PhotoImage(volume_button_img)

    except OSError:
        print("Error while loading images...")
        if play_button_img:
            play_button_img.close()
        if pause_button_img:
            pause_button_img.close()
        if volume_button_img:
            volume_button_img.close()

        sys.exit()

    else:
        return play_button, pause_button, volume_button


class CustomProgressBar(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.master = master
        self.images_dir: str = 'GUI/images'
        self.play_button: tk.PhotoImage = ...
        self.pause_button: tk.PhotoImage = ...
        self.volume_button: tk.PhotoImage = ...

        self.play_button, self.pause_button, self.volume_button = (
            load_buttons_images(self.images_dir))

        self.btn_play_video: tk.Button = tk.Button(self, image=str(self.play_button),
                                                   relief="flat", height=20, width=20,
                                                   command=self.master.check_play_button,
                                                   bg='light grey')

        self.btn_video_volume: tk.Button = tk.Button(self, image=str(self.volume_button),
                                                     relief="flat", height=20, width=20, bg='light grey')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", troughcolor='light grey', background='green',
                        borderwidth=3, bordercolor='gray')

        self.progress_bar: ttk.Progressbar = ttk.Progressbar(self, orient='horizontal',
                                                             mode='determinate', style='TProgressbar')

        self.rowconfigure(0, weight=0)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.btn_play_video.grid(row=0, column=0, sticky="nsew")
        self.btn_video_volume.grid(row=0, column=2, sticky="e")
        self.progress_bar.grid(row=0, column=1, sticky="ew")

        self.progress_bar.bind("<Button-1>", self.master.seek_video)

    def __del__(self):
        self.play_button = None
        self.volume_button = None
        self.pause_button = None

    def update_progress_bar(self, current_frames: float) -> None:
        if self.progress_bar['value'] < self.progress_bar["length"]:
            self.progress_bar['value'] = (current_frames / self.progress_bar["length"]) * 100

    def button_pressed(self):
        if self.btn_play_video["image"] is not None:
            if self.btn_play_video["image"] == str(self.play_button):
                self.btn_play_video["image"] = self.pause_button
                return True
            else:
                self.btn_play_video["image"] = self.play_button
                return False

    def reset_progress_bar(self, current_frames: float) -> None:
        self.progress_bar['value'] = 0
        self.progress_bar['length'] = current_frames
        self.btn_play_video["image"] = self.play_button
