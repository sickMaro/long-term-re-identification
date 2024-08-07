import sys

from window import Window

sys.path.append('../methods/SOLIDER-REID')
from integration_file import parse_args

if __name__ == "__main__":
    cfg = parse_args()
    window = Window(cfg)
    window.show_window()
