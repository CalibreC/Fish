import sys
import cv2
sys.path.append("../yolov5-7.0/")
from models.experimental import attempt_load

import torch
import torchvision


def try_gpu(i=0):
    """If GPU is available, return torch.device as cuda:i; else return torch.device as cpu."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


device = try_gpu(0)


# 载入模型
def load_model():
    model = attempt_load("../weights/2023-08-02.pt", device=device)
    return model


# 推理
def inference(model, img):
    img = torch.from_numpy(img).cuda()
    with torch.no_grad():
        prediction = model([img])
    return prediction


# 后处理
def postprocess(prediction):
    return prediction



if __name__ == '__main__':
    model = load_model()
    img = cv2.imread("1.png")
    image = torch.tensor(img)
    results = model(image)
    print(len(results))
    # postprocess(prediction)
