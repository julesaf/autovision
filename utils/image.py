import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def visualize_bboxes(img, boxes, labels, figsize=(12, 10)):
    box = draw_bounding_boxes(
        torch.Tensor(img).type(torch.uint8),
        boxes=torch.Tensor(boxes),
        labels=labels,
        colors="red",
        width=4,
        font_size=30
    )
    plt.figure(figsize=figsize)
    plt.imshow(to_pil_image(box))
    plt.show()


def show_img(img, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()