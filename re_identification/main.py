from GUI import MainWindow
from parser import parse_args
from identification import ReIdentificationManager
from video_manager import VideoManager

if __name__ == "__main__":
    cfg = parse_args()
    video_manager = VideoManager()
    re_id_manager = ReIdentificationManager(cfg)
    re_id_manager.load_models()
    mw = MainWindow(cfg, video_manager, re_id_manager)
    mw.mainloop()
