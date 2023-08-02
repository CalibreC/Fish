import time

import cv2
import dxcam
from PIL import Image

region = (0, 0, 2560, 1440)


def screenshot():
    """
    create 参数:
        device_idx: int = 0,            # GPU
        output_idx: int = None,         # Monitor
        region: tuple = None,           # (x, y, w, h)
        output_color: str = "RGB",      # support RGB, BGR, RGBA, BGRA, GRAY
        max_buffer_len: int = 64,       # 最大缓存帧数
    """
    camera = dxcam.create()  # returns a DXCamera instance on primary monitor
    # camera = dxcam.create(output_color="BGRA")

    frame = camera.grab(region=region)

    Image.fromarray(frame).show()

    """
    Opencv需要反转颜色,dxcam.create时设置output_color="BGRA"后不需要反转
    """
    # cv2.imshow('test.png', frame)
    # cv2.waitKey(0)

    del camera


def screen_capture():
    """
    start 参数:
        region: Tuple[int, int, int, int] = None,
        target_fps: int = 60,
        video_mode=False,
        delay: int = 0,
    """

    camera = dxcam.create()

    # Start capturing video
    # video_mode = True capture not block
    camera.start(video_mode=True, region=region)
    for i in range(1000):
        print(camera.is_capturing)  # True
        image = camera.get_latest_frame()  # Will block until new frame available
    camera.stop()
    del camera


def multiple_monitors():
    # cross GPU untested, Nvidia MX150 + Intel UHD 620 cannot work together
    # device means GPU, output means monitor
    cam1 = dxcam.create(device_idx=0, output_idx=0)
    cam2 = dxcam.create(device_idx=0, output_idx=1)
    # cam3 = dxcam.create(device_idx=1, output_idx=0)
    img1 = cam1.grab()
    img2 = cam2.grab()
    # img2 = cam3.grab()
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)

    del cam1
    del cam2


def max_fps():
    title = "max fps"
    start_time, fps = time.perf_counter(), 0
    cam = dxcam.create()
    start = time.perf_counter()
    while fps < 1000:
        frame = cam.grab()
        if frame is not None:  # New frame
            fps += 1
    end_time = time.perf_counter() - start_time
    print(f"{title}: {fps / end_time}")


if __name__ == "__main__":
    # screenshot()
    # screen_capture()
    # multiple_monitors()
    max_fps()
