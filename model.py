import torch.nn as nn
import torchvision


def build_model(num_classes: int = 2):
    """
    Pretrained R(2+1)D-18 on Kinetics-400, fine-tuned for 2 classes.
    """
    weights = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
    model = torchvision.models.video.r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
