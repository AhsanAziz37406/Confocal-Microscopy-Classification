import timm
import torch.nn as nn


def get_model(num_classes):
    model = timm.create_model("maxvit_large_tf_512.in1k", pretrained=True)
    model.reset_classifier(num_classes)  # Adjust final layer
    return model
