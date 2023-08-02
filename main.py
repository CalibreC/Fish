import time

import cv2
import dxcam
from PIL import Image

region = (0, 0, 2560, 1440)


def screenshot():
    camera = dxcam.create()  # returns a DXCamera instance on primary monitor

    frame = camera.grab(region=region)  # numpy.ndarray of size (640x640x3) -> (HXWXC)

    Image.fromarray(frame).show()
    # cv2.imwrite('screenshot1920.png', frame)          # Opencv需要反转颜色


def screen_capture():
    camera = dxcam.create()

    camera.start(video_mode=True, region=region)  # Start capturing video
    for i in range(10):
        print(camera.is_capturing)  # True
        image = camera.get_latest_frame()  # Will block until new frame available
    camera.stop()


if __name__ == "__main__":
    # screenshot()
    screen_capture()
