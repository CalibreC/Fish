import torch


def load_model(
    device="cuda:0",
    weights=".\\weights\\2023-08-07.pt",
):
    ckpt = torch.load(weights, map_location=device)  # load
    ckpt = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model
    fuse = True
    model = (
        ckpt.fuse().eval() if fuse else ckpt.eval()
    )  # fused or un-fused model in eval mode
    model.float()
    class_names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names
    return model, class_names
