import tkinter as tk


class MainWindow(tk.Tk):

    def __init__(self, cfg, video_manager, re_id_manager, **kw):
        from camera_section import CameraSection
        from video_section import VideoSection, ProbeSection
        super().__init__(**kw)
        self.MAIN_TITLE: str = 'Person re-identification - SATcase6video'
        self.RESULT_TITLE: str = 'Result - SATcase6video'
        self.cfg = cfg
        self.video_manager = video_manager
        self.re_id_manager = re_id_manager
        self.configure_window(main=True)
        self.results: list[tk.PhotoImage] = []

        self.frm_main: tk.Frame = tk.Frame(self, bg="light grey")

        self.lbl_title: tk.Label = tk.Label(self, fg="black", bg="light grey",
                                            text=self.MAIN_TITLE, height=2,
                                            relief="raised",
                                            font=("Arial", 25))

        self.frm_video = VideoSection(self, self.video_manager, self.re_id_manager,
                                      bg="light grey")

        self.frm_probe = ProbeSection(self, bg="light grey")

        self.frm_cameras = CameraSection(self, self.frm_video, self.cfg.DATASETS.DAY,
                                         bg="light grey")

        self.frm_main.rowconfigure((0, 1), weight=0)
        self.frm_main.columnconfigure(0, weight=1)
        self.frm_main.columnconfigure(1, weight=0)
        # self.frm_main.grid(row=1, column=0, sticky="new", padx=(15, 15), pady=35)

        self.lbl_title.grid(row=0, column=0, columnspan=2, sticky="new", padx=2, pady=2)

        self.frm_video.grid(row=1, column=0, sticky="new", padx=(15, 15), pady=35)
        self.frm_cameras.grid(row=1, column=1, sticky="nsew", padx=(0, 15), pady=35)

        self.frm_probe.grid(row=1, column=0, sticky="new", padx=(15, 15), pady=35)
        self.frm_probe.grid_remove()


    def configure_window(self, main: bool = False, multiplier: float = 1.5) -> None:
        if main:
            self.state('zoomed')
            self.resizable(True, True)

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, minsize=self.winfo_screenmmwidth(), weight=1)
        self.columnconfigure(1, minsize=self.winfo_screenmmwidth() * multiplier, weight=1)

    def change_main_view(self, future):
        results = future.result()
        self.lbl_title.config(text=self.RESULT_TITLE)
        self.configure_window(multiplier=1.7)
        self.frm_video.grid_remove()
        self.frm_probe.grid()
        self.frm_probe.lbl_probe.config(image=self.video_manager.get_current_selected_area())
        self.frm_cameras.display_results_view(results)
        # self.after(100, self.__create_results_frames, results)

    def restore_main_view(self):
        self.lbl_title.config(text=self.MAIN_TITLE)
        self.configure_window()
        self.frm_probe.grid_remove()
        self.frm_video.grid()
        self.frm_video.btn_identification.config(state="normal")
        self.frm_cameras.display_camera_view()
