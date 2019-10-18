import torch
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt


def get_metric(metric, metrics, mode):
    return list(map(lambda x: x[metric], metrics[mode]))


def plot_metrics(metrics, mode):

    loss = get_metric("loss", metrics, mode)
    plt.plot(loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()

    plt.figure(figsize=(12, 10))
    samplewise_accuracy = get_metric("samplewise_accuracy", metrics, mode)
    plt.subplot(2, 2, 1)
    plt.plot(samplewise_accuracy)
    plt.ylabel("Samplewise accuracy")
    plt.xlabel("Epoch")

    mean_accuracy = get_metric("mean_accuracy", metrics, mode)
    plt.subplot(2, 2, 2)
    plt.plot(mean_accuracy)
    plt.ylabel("Mean accuracy")
    plt.xlabel("Epoch")

    mean_iou = get_metric("mean_iou", metrics, mode)
    plt.subplot(2, 2, 3)
    plt.plot(mean_iou)
    plt.ylabel("Mean IoU")
    plt.xlabel("Epoch")

    frequency_weighted_iou = get_metric("frequency_weighted_iou", metrics, mode)
    plt.subplot(2, 2, 4)
    plt.plot(frequency_weighted_iou)
    plt.ylabel("Frequency weighted IoU")
    plt.xlabel("Epoch")

    plt.show()


def predict_with_model(model, dataset, device):
    preds = []
    gts = []
    for i in range(len(dataset)):
        data, label = dataset[i]
        preds.extend(
            model(data.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float))
            .argmax(1)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        gts.extend(label["y"].numpy())

    preds = np.array(preds)
    gts = np.array(gts)

    return gts, preds


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.grid(False)
    fig.tight_layout()
    return ax
