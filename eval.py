import json
import matplotlib.pyplot as plt
import numpy as np
files = [
    "layer5scaffold.json",
    "layer7fedavg.json",
    "layer5fedavg.json",
    "layer7scaffold.json",
]

fig, ax = plt.subplots(4)

for i, fname in enumerate(files):
    with open("logs/"+fname, "r") as f:
        data = json.load(f)

    train_losses = np.asarray(data["train_losses"])
    min_loss = train_losses.mean(axis = 0).min()
    ax[i].plot(range(train_losses.shape[1]), train_losses.mean(axis = 0))
    print(f"{fname}: min train loss = {min_loss}")
fig.savefig("check.jpg")
