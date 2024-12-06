import json
from PIL import Image, ImageDraw
from dataclasses import dataclass
import torch
import numpy as np
from pathlib import Path


def get_json_path(json_name, data_dir):
    json_path = data_dir.joinpath(json_name)
    return json_path


@dataclass
class ImgsData:
    """
    A data class to store cropped images grouped by labels.
    """

    img: any
    RSJ_imgs: list
    LSJ_imgs: list
    RHIP_imgs: list
    LHIP_imgs: list


def padding_img(image):
    width, height = image.size

    max_side = max(width, height)

    new_image = Image.new("RGB", (max_side, max_side), (0, 0, 0))

    new_image.paste(
        image, ((max_side - width) // 2, (max_side - height) // 2)
    )

    resized_image = new_image.resize((224, 224))

    return resized_image

def read_json_with_encoding(file_path):
    
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            print(f" {encoding} Failed, Try another...")
    raise UnicodeDecodeError(f"无法使用以下编码读取文件：{encodings}")

def crop_pelvic_img_from_json(example, data_dir):
    # print(example)
    json_path = get_json_path(example["json_name"], data_dir)
    json_data = read_json_with_encoding(json_path)
    # decode json data
    data = json.loads(json_data)
    img = example["image"]
    resize_img = padding_img(img)

    RSJ_imgs = []
    LSJ_imgs = []
    RHIP_imgs = []
    LHIP_imgs = []

    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        right = max(point[0] for point in points)
        bottom = max(point[1] for point in points)

        if label == "RSJ":
            RSJ_imgs.append(
                padding_img(img.crop((left, top, right, bottom)))
            )
        if label == "LSJ":
            LSJ_imgs.append(
                padding_img(img.crop((left, top, right, bottom)))
            )
        if label == "RHIP":
            RHIP_imgs.append(
                padding_img(img.crop((left, top, right, bottom)))
            )
        if label == "LHIP":
            LHIP_imgs.append(
                padding_img(img.crop((left, top, right, bottom)))
            )

    crop_imgs = ImgsData(
        resize_img, RSJ_imgs, LSJ_imgs, RHIP_imgs, LHIP_imgs
    )

    return crop_imgs


def get_batch_img(examples, data_dir):
    imgs = []
    RSJ_imgs = []
    LSJ_imgs = []
    RHIP_imgs = []
    LHIP_imgs = []
    for example in examples:
        crop_imgs = crop_pelvic_img_from_json(example, data_dir)
        img = crop_imgs.img
        RSJ_img = crop_imgs.RSJ_imgs[0]
        LSJ_img = crop_imgs.LSJ_imgs[0]
        RHIP_img = crop_imgs.RHIP_imgs[0]
        LHIP_img = crop_imgs.LHIP_imgs[0]
        imgs.append(img)
        RSJ_imgs.append(RSJ_img)
        LSJ_imgs.append(LSJ_img)
        RHIP_imgs.append(RHIP_img)
        LHIP_imgs.append(LHIP_img)
    return ImgsData(imgs, RSJ_imgs, LSJ_imgs, RHIP_imgs, LHIP_imgs)


def preprocess_imgs_batch(
    examples, data_dir, train_transforms=None, val_transforms=None
):
    batch = get_batch_img(examples, data_dir)
    imgs_batch = batch.img
    RSJ_batch = batch.RSJ_imgs
    LSJ_batch = batch.LSJ_imgs
    RHIP_batch = batch.RHIP_imgs
    LHIP_batch = batch.LHIP_imgs
    if examples[0]["json_name"].split("/")[0] == "train":
        # print("train batch")
        transformed_imgs_batch = [
            train_transforms(image.convert("RGB"))
            for image in imgs_batch
        ]
        transformed_RSJ_batch = [
            train_transforms(image.convert("RGB"))
            for image in RSJ_batch
        ]
        transformed_LSJ_batch = [
            train_transforms(image.convert("RGB"))
            for image in LSJ_batch
        ]
        transformed_RHIP_batch = [
            train_transforms(image.convert("RGB"))
            for image in RHIP_batch
        ]
        transformed_LHIP_batch = [
            train_transforms(image.convert("RGB"))
            for image in LHIP_batch
        ]

    if examples[0]["json_name"].split("/")[0] == "val":
        # print("validation batch")
        transformed_imgs_batch = [
            val_transforms(image.convert("RGB"))
            for image in imgs_batch
        ]
        transformed_RSJ_batch = [
            val_transforms(image.convert("RGB"))
            for image in RSJ_batch
        ]
        transformed_LSJ_batch = [
            val_transforms(image.convert("RGB"))
            for image in LSJ_batch
        ]
        transformed_RHIP_batch = [
            val_transforms(image.convert("RGB"))
            for image in RHIP_batch
        ]
        transformed_LHIP_batch = [
            val_transforms(image.convert("RGB"))
            for image in LHIP_batch
        ]
    return ImgsData(
        torch.stack(transformed_imgs_batch).float().to("cuda"),
        torch.stack(transformed_RSJ_batch).float().to("cuda"),
        torch.stack(transformed_LSJ_batch).float().to("cuda"),
        torch.stack(transformed_RHIP_batch).float().to("cuda"),
        torch.stack(transformed_LHIP_batch).float().to("cuda"),
    )


def preprocess_labels_batch(examples):
    labels_list = []
    for example in examples:
        num_list = [float(x) for x in example["labels"].split(",")]
        np_array = np.array(num_list)
        tensors = torch.tensor(np_array)
        labels_list.append(tensors)
    labels = torch.stack(
        labels_list,
    )
    return labels.float().to("cuda")
