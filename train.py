import os
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import defaultdict

from lcd.dataset import CrossTripletDataset
from lcd.models import *
from lcd.losses import *


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="path to the json config file")
parser.add_argument("--logdir", help="path to the log directory")
args = parser.parse_args()

config = args.config
logdir = args.logdir
args = json.load(open(config))

if not os.path.exists(logdir):
    os.mkdir(logdir)

fname = os.path.join(logdir, "config.json")
with open(fname, "w") as fp:
    json.dump(args, fp, indent=4)

device = args["device"]

dataset = CrossTripletDataset(args["root"], split="train")
loader = data.DataLoader(
    dataset,
    batch_size=args["batch_size"],
    num_workers=args["num_workers"],
    pin_memory=True,
    shuffle=True,
)

patchnet = PatchNetAutoencoder(args["embedding_size"], args["normalize"])
pointnet = PointNetAutoencoder(
    args["embedding_size"],
    args["input_channels"],
    args["output_channels"],
    args["normalize"],
)
patchnet.to(device)
pointnet.to(device)

parameters = list(patchnet.parameters()) + list(pointnet.parameters())
optimizer = optim.SGD(
    parameters,
    lr=args["learning_rate"],
    momentum=args["momentum"],
    weight_decay=args["weight_decay"],
)

criterion = {
    "mse": MSELoss(),
    "chamfer": ChamferLoss(args["output_channels"]),
    "triplet": HardTripletLoss(args["margin"], args["hardest"]),
}
criterion["mse"].to(device)
criterion["chamfer"].to(device)
criterion["triplet"].to(device)

best_loss = np.Inf
for epoch in range(args["epochs"]):
    start = datetime.datetime.now()
    scalars = defaultdict(list)

    for i, batch in enumerate(loader):
        x = [x.to(device) for x in batch]
        y0, z0 = pointnet(x[0])
        y1, z1 = patchnet(x[1])

        loss_r = 0
        loss_d = 0
        loss_r += args["alpha"] * criterion["mse"](x[1], y1)
        loss_r += args["beta"] * criterion["chamfer"](x[0], y0)
        loss_d += args["gamma"] * criterion["triplet"](z0, z1)
        loss = loss_d + loss_r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scalars["loss"].append(loss)
        scalars["loss_d"].append(loss_d)
        scalars["loss_r"].append(loss_r)

        now = datetime.datetime.now()
        log = "{} | Batch [{:04d}/{:04d}] | loss: {:.4f} |"
        log = log.format(now.strftime("%c"), i, len(loader), loss.item())
        print(log)

    # Summary after each epoch
    summary = {}
    now = datetime.datetime.now()
    duration = (now - start).total_seconds()
    log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
    log = log.format(now.strftime("%c"), epoch, args["epochs"], duration)
    for m, v in scalars.items():
        summary[m] = torch.stack(v).mean()
        log += " {}: {:.4f} |".format(m, summary[m].item())

    fname = os.path.join(logdir, "checkpoint_{:04d}.pth".format(epoch))
    print("> Saving model to {}...".format(fname))
    model = {"pointnet": pointnet.state_dict(), "patchnet": patchnet.state_dict()}
    torch.save(model, fname)

    if summary["loss"] < best_loss:
        best_loss = summary["loss"]
        fname = os.path.join(logdir, "model.pth")
        print("> Saving model to {}...".format(fname))
        model = {"pointnet": pointnet.state_dict(), "patchnet": patchnet.state_dict()}
        torch.save(model, fname)
    log += " best: {:.4f} |".format(best_loss)

    fname = os.path.join(logdir, "train.log")
    with open(fname, "a") as fp:
        fp.write(log + "\n")

    print(log)
    print("--------------------------------------------------------------------------")
