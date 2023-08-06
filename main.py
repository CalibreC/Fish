#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/3 1:43
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : capture.py
# @Description      :
import argparse
import sys
import time
from ctypes import windll

import cv2
import dxcam
import numpy as np
import torch
import torchvision
import win32gui
import win32ui
import yaml
from loguru import logger
from PIL import Image

from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device

# window_name = "崩坏：星穹铁道"
window_name = "原神"
# window_name = "向日葵远程控制"
# window_name = "Notepad"
# capture_method = "win32api"
capture_method = "dxcam"


def get_window_info():
    """

    :return: 窗口句柄，左上角坐标，右下角坐标
    """
    # FindWindow(class-name, window-name)
    # hwnd = win32gui.FindWindow(window_name, None)
    hwnd = win32gui.FindWindow("UnityWndClass", window_name)
    if hwnd is None:
        logger.error("未找到窗口")
        sys.exit(1)

    # 如果使用高 DPI 显示器（或 > 100% 缩放尺寸），添加下面一行，否则注释掉
    windll.user32.SetProcessDPIAware()

    # 根据您是想要整个窗口还是只需要 client area 来更改下面的行。
    client = win32gui.GetClientRect(hwnd)
    logger.info(f"client: {client}")
    window = win32gui.GetWindowRect(hwnd)
    logger.info(f"window: {window}")

    left = client[0]
    top = client[1]
    right = client[2]
    bot = client[3]

    # left, top, right, bot = client[0], client[1], client[2], client[3]

    if capture_method == "win32api":
        pass
    elif capture_method == "dxcam":
        window_width = window[2] - window[0]
        window_height = window[3] - window[1]
        client_width = client[2] - client[0]
        client_height = client[3] - client[1]
        align = (window_width - client_width) / 2
        left = int(window[0] + align)
        right = int(window[2] - align)
        bot = int(window[3] - align)
        top_align = window_height - client_height - align
        top = int(window[1] + top_align)
    else:
        raise ValueError("capture_method should be win32api or dxcam")

    logger.info(f"left: {left}, top: {top}, right: {right}, bot: {bot}")
    return hwnd, (left, top, right, bot)


def dxcam_init():
    camera = dxcam.create(
        device_idx=0, output_color="BGRA", output_idx=0
    )  # returns a DXCamera instance on primary monitor
    return camera


camera = dxcam_init()


def dxcam_capture():
    hwnd, (left, top, right, bot) = get_window_info()

    im = camera.grab(region=(left, top, right, bot))

    if im is not None:
        # PrintWindow Succeeded
        # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
        return im  # 返回图片
    else:
        logger.error("截图失败")
        return None


def win32api_capture():
    hwnd, (left, top, right, bot) = get_window_info()

    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)  # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)  # 根据窗口的DC获取mfcDC
    saveDC = mfcDC.CreateCompatibleDC()  # mfcDC创建可兼容的DC

    saveBitMap = win32ui.CreateBitmap()  # 创建bitmap准备保存图片
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)  # 为bitmap开辟空间

    saveDC.SelectObject(saveBitMap)  # 高度saveDC，将截图保存到saveBitmap中

    # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST,
    # 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    # 选择合适的 window number，如0，1，2，3，直到截图从黑色变为正常画面
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
    # win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST,
    # 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        "RGB",
        (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
        bmpstr,
        "raw",
        "BGRX",
        0,
        1,
    )

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    # if result == 1:
    # PrintWindow Succeeded
    # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
    return im  # 返回图片
    # else:
    #     logger.error("截图失败")
    #     return None


def video_capture(window_name, capture_method):
    if capture_method == "win32api":
        capture = win32api_capture
    elif capture_method == "dxcam":
        capture = dxcam_capture
    else:
        raise ValueError("capture_method should be win32api or dxcam")

    while True:
        start = time.time()
        img = capture()
        if img is not None:
            image = np.asarray(img)
            # img = torch.from_numpy(img).cuda()
            # 推理，后处理
            cv2.imshow("image", image)
            cv2.waitKey(1)
        else:
            time.sleep(0.04)
        end = time.time()
        fps = 1 / np.round(end - start, 3)
        print(f"Frames Per Second : {fps}")


def load_model(
    device="0",
    weights=".\\weights\\2023-08-02.pt",
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=".\\data\\fish.yaml")
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((1920, 1080), s=stride)  # check image size


def make_parser():
    parser = argparse.ArgumentParser("Capture")
    parser.add_argument("--name", default="原神", type=str, help="选择合适的")
    parser.add_argument(
        "--capture_method", default="dxcam", type=str, help="train or test"
    )
    return parser


def letterbox(
    im,
    new_shape=(1920, 1088),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def preprocess(cv_image, img_size=(544, 960)):
    cv_image = np.asarray(cv_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)

    # Padded resize
    temp_image = letterbox(cv_image, new_shape=img_size)[0]

    # numpy to tensor
    image = torch.from_numpy(temp_image).cuda()

    # Convert
    # HWC to CHW
    image = image.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return image, cv_image


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    # where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
):
    """
    pred: 网络的输出结果
    conf_thres:置信度阈值
    ou_thres:iou阈值
    classes: 是否只保留特定的类别
    agnostic_nms: 进行nms是否也去除不同类别之间的框
    max-det: 保留的最大检测框数量
    ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
    pred是一个列表list[torch.tensor], 长度为batch_size
    每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
    """
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = (
        2,
        7680,
    )  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[
            ((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4
        ] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = (
            x[:, :4] + c,
            x[:, 4],
        )  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def yaml_load(file="data.yaml"):
    # Single-line safe yaml loading
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def box_label(image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width (pixels)

    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def postprocess(pred, image, cv_image):
    gn = torch.tensor(cv_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                image.shape[2:], det[:, :4], cv_image.shape
            ).round()
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                box_label(
                    cv_image,
                    box=xyxy,
                    label=label,
                    color=(128, 128, 128),
                    txt_color=(255, 255, 255),
                )
                # plot_one_box(xyxy, cv_image, label=label, color=colors[int(cls)], line_thickness=3)

    return cv_image


if __name__ == "__main__":
    logger.remove()  # 删除自动产生的handler
    handle_id = logger.add(sys.stderr, level="WARNING")  # 添加一个可以修改控制的handler
    args = make_parser().parse_args()

    # load_model()
    # data_path = ".\\data\\fish.yaml"
    # class_names = yaml_load(data_path)['names'] if data_path else {i: f'class{i}' for i in range(999)}

    device = select_device("0")
    model = DetectMultiBackend(
        ".\\weights\\2023-08-07.pt", device=device, data=".\\data\\fish.yaml"
    )
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((1920, 1080), s=stride)  # check image size

    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    if capture_method == "win32api":
        capture = win32api_capture
    elif capture_method == "dxcam":
        capture = dxcam_capture
    else:
        raise ValueError("capture_method should be win32api or dxcam")

    while True:
        start = time.time()
        img = capture()
        if img is not None:
            image, cv_image = preprocess(img)

            # inference
            """
            pred.shape=(1, num_boxes, 5+num_class)
            h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
            num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
            pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
            pred[..., 4]为objectness置信度
            pred[..., 5:-1]为分类结果
            """
            pred = model(image, augment=False, visualize=False)[0]

            # NMS
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

            # postprocess
            cv_image = postprocess(pred, image, cv_image)

            cv2.imshow("image", cv_image)
            cv2.waitKey(1)
        else:
            time.sleep(0.04)
        end = time.time()
        fps = 1 / np.round(end - start, 3)
        print(f"Frames Per Second : {fps}")
