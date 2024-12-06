import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-label image classification model.")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="google/vit-base-patch16-224",
        help="Model checkpoint from Hugging Face model hub",
    )
    parser.add_argument(
        "--trainvaldataset_dir",
        type=str,
        default="../data/cv_datasets/train12_val3",
        help="Dataset dir",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    return parser.parse_args()


my_args = parse_args()

import comet_ml

# Log in to Comet (if necessary)
comet_ml.login()


experiment_config = comet_ml.ExperimentConfig(
    name=my_args.model_checkpoint + "-" + my_args.trainvaldataset_dir,
    tags=["cv_training"],
)

experiment = comet_ml.start(
    project_name="comet-example-start-logging",
    experiment_config=experiment_config,
)

from pathlib import Path
from datasets import Image, load_dataset

data_dir = Path(my_args.trainvaldataset_dir)

dataset = load_dataset("imagefolder", data_dir=data_dir)
dataset = dataset.cast_column("image", Image(mode="RGB"))

 
labels = ["diagnosis", "RSJ", "LSJ", "RHIP", "LHIP"]
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

from transformers import AutoImageProcessor

model_checkpoint = my_args.model_checkpoint
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
image_processor

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (
        image_processor.size["height"],
        image_processor.size["width"],
    )
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
    [
        RandomResizedCrop(crop_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        normalize,
    ]
)

train_ds = dataset["train"]
val_ds = dataset["validation"]

from utils import (
    preprocess_imgs_batch,
    preprocess_labels_batch,
    ImgsData,
)

import numpy as np
import torch


def collate_fn(examples):
    imgs_batch = preprocess_imgs_batch(examples, data_dir, train_transforms, val_transforms)
    labels = preprocess_labels_batch(examples)
    return {"imgs_batch": imgs_batch, "labels": labels}


from transformers import AutoModelForImageClassification
import torch.nn as nn

# vit-base
if my_args.model_checkpoint == "google/vit-base-patch16-224":
    from models import CustomVitForMultiLabelImageClassification

    AutoModelForImageClassification = CustomVitForMultiLabelImageClassification

# poolformer_m48
if my_args.model_checkpoint == "sail/poolformer_m48":
    from models import CustomPoolformerForMultiLabelImageClassification

    AutoModelForImageClassification = CustomPoolformerForMultiLabelImageClassification

# convnext
if my_args.model_checkpoint == "facebook/convnext-base-224":
    from models import CustomConvnextForMultiLabelImageClassification

    AutoModelForImageClassification = CustomConvnextForMultiLabelImageClassification

# convnextv2
if my_args.model_checkpoint == "facebook/convnextv2-base-1k-224":
    from models import CustomConvnextv2ForMultiLabelImageClassification

    AutoModelForImageClassification = CustomConvnextv2ForMultiLabelImageClassification

# resnet50
if my_args.model_checkpoint == "microsoft/resnet-50":
    from models import CustomResNetForMultiLabelImageClassification

    AutoModelForImageClassification = CustomResNetForMultiLabelImageClassification

# Swin-base
if my_args.model_checkpoint == "microsoft/swin-base-patch4-window7-224":
    from models import CustomSwinForMultiLabelImageClassification

    AutoModelForImageClassification = CustomSwinForMultiLabelImageClassification


import types

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
model.attention_weights = nn.Parameter(torch.ones(5))

model.method = "attention"  # 或 'attention'
model.forward = types.MethodType(AutoModelForImageClassification.forward, model)

model = model.to("cuda")

from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import numpy as np


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # 检查并解包 logits
    if isinstance(logits, tuple):
        logits = logits[0]  

    print(logits.shape)
    # print(labels)

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    predictions = (logits > 0.5).astype(int)

    accuracy = accuracy_score(labels, predictions)
    hamming_accuracy = 1 - hamming_loss(labels, predictions)
    f1 = f1_score(labels, predictions, average="samples")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "hamming_accuracy": hamming_accuracy,
    }


from transformers import Trainer, TrainingArguments

model_name = model_checkpoint.split("/")[-1]
fold = my_args.trainvaldataset_dir.split("/")[-1]
batch_size = my_args.batch_size  
args = TrainingArguments(
    f"output/{model_name}-finetuned-eurosat/{fold}",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    dataloader_pin_memory=False,
    save_total_limit=2,
    learning_rate=5e-5,
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()
