import os
import time
import yaml
import copy
from pathlib import Path
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import models
import ops.trains as trains
import ops.tests as tests
import ops.datasets as datasets
import ops.schedulers as schedulers

import timm


root = "."

# config_path = "%s/configs/cifar10_vit.yaml" % root
config_path = "%s/configs/cifar100_vit.yaml" % root
# config_path = "%s/configs/imagenet_vit.yaml" % root

with open(config_path) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)



dataset_args = copy.deepcopy(args).get("dataset")
train_args = copy.deepcopy(args).get("train")
val_args = copy.deepcopy(args).get("val")
model_args = copy.deepcopy(args).get("model")
optim_args = copy.deepcopy(args).get("optim")
env_args = copy.deepcopy(args).get("env")

dataset_train, dataset_test = datasets.get_dataset(**dataset_args, download=True)
dataset_name = dataset_args["name"]
num_classes = len(dataset_train.classes)

# total_length = len(dataset_train)
# train_length = int(0.1 * total_length)  # 50% for training
# dataset_train = random_split(dataset_train, [train_length, total_length - train_length])[0]

dataset_train = DataLoader(dataset_train, 
                           shuffle=True, 
                           num_workers=train_args.get("num_workers", 4), 
                           batch_size=train_args.get("batch_size", 128))
dataset_test = DataLoader(dataset_test, 
                          num_workers=val_args.get("num_workers", 4), 
                          batch_size=val_args.get("batch_size", 128))




print("Train: %s, Test: %s, Classes: %s" % (
    len(dataset_train.dataset), 
    len(dataset_test.dataset), 
    num_classes
))

blocks = {
    "tiny": timm.models.vision_transformer.Block,
    "eigenpos": timm.models.vision_transformer.BlockEigenPos,
    "eigenneg": timm.models.vision_transformer.BlockEigenNeg
}

for m in ["eigenpos","eigenneg","tiny"]:
    for d in [24,48]:
        model = timm.models.vision_transformer.VisionTransformer(
            num_classes=num_classes, img_size=32, patch_size=2,  # for CIFAR
            embed_dim=192, depth=d, num_heads=3, qkv_bias=False,  # for ViT-Ti 
            block_fn=blocks[m]
        )

        model.name = "vit_"+m
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        name = model.name
        model = nn.DataParallel(model)
        model.name = name

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("runs", dataset_name, model.name, current_time)
        writer = SummaryWriter(log_dir)

        with open("%s/config.yaml" % log_dir, "w") as f:
            yaml.dump(args, f)
        with open("%s/model.log" % log_dir, "w") as f:
            f.write(repr(model))

        print("Create TensorBoard log dir: ", log_dir)

        gpu = torch.cuda.is_available()
        optimizer, train_scheduler = trains.get_optimizer(model, **optim_args)
        warmup_scheduler = schedulers.WarmupScheduler(optimizer, len(dataset_train) * train_args.get("warmup_epochs", 0))

        trains.train(model, optimizer,
                    dataset_train, dataset_test,
                    train_scheduler, warmup_scheduler,
                    train_args, val_args, gpu,
                    writer, 
                    snapshot=10, dataset_name=dataset_name, uid=current_time)  # Set `snapshot=N` to save snapshots every N epochs.
        
        models.save(model, dataset_name, current_time, optimizer=optimizer)

        gpu = torch.cuda.is_available()

        model = model.cuda() if gpu else model.cpu()
        metrics_list = []
        for n_ff in [1]:
            print("N: %s, " % n_ff, end="")
            *metrics, cal_diag = tests.test(model, n_ff, dataset_test, verbose=False, gpu=gpu)
            metrics_list.append([n_ff, *metrics])

        leaderboard_path = os.path.join("leaderboard", "logs", dataset_name, model.name)
        Path(leaderboard_path).mkdir(parents=True, exist_ok=True)
        metrics_dir = os.path.join(leaderboard_path, "%s_%s_%s.csv" % (dataset_name, model.name, current_time))
        tests.save_metrics(metrics_dir, metrics_list)
