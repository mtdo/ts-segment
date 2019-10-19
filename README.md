# ts-segment

ts-segment is a Python library for creating semantic segmentation models for multivariate time series, primarily (but not exclusively) for motion sensor data.

The library is based on the PyTorch deep learning framework and on the Ignite package which helps write compact and full-featured training loops with just a few lines of code, supporting metrics, early-stopping, model checkpointing, learning rate scheduling and more.

## Getting started
The ts-segment library is compatible with Ignite and follows the same core concepts, so to get started with ts-segment all you need is to define an iterator class for your dataset under `segmentation/datasets/`, load a model, create a training engine with any standard PyTorch optimizer and loss function, and run the engine on your data.
```
from segmentation.datasets import YourDataset
from segmentation.models import Model
from segmentation.engine import create_trainer

dataset = YourDataset()
model = Model()
trainer = create_trainer(model, optimizer, loss_fn)
trainer.run(dataset)
```

## Time series segmentation metrics
The ts-segment library allows easy logging of the most common semantic segmentation metrics including samplewise accuracy, mean accuracy, mean IoU and frequency weighted IoU. To log metrics during training, create an evaluator engine with your metrics, create a logger object with the train and validation datasets and attach it to your training engine.

```
from ignite.engine import Events
from segmentation.engine import create_trainer, create_evaluator
from segmentation.metrics import (
    SamplewiseAccuracy,
    MeanAccuracy,
    MeanIoU,
    FrequencyWeightedIoU,
)
from segmentation.logger import Logger

trainer = create_trainer(model, optimizer, loss_fn)
evaluator = create_evaluator(
    model,
    metrics={
        "loss": Loss(loss_fn),
        "samplewise_accuracy": SamplewiseAccuracy(),
        "mean_accuracy": MeanAccuracy(),
        "mean_iou": MeanIoU(),
        "frequency_weighted_iou": FrequencyWeightedIoU(),
    },
)

logger = Logger(evaluator, train_dataset, validation_dataset)
trainer.add_event_handler(Events.EPOCH_COMPLETED, logger)

trainer.run(dataset)
```
After training finishes, the logger object contains a dictionary of epoch-wise metrics.

## Example
An example application is included in this repository under `example/` where ts-segment is used to segment time series of motion sensor data for activity recognition. Before running the training notebook, the data should first be downloaded and then prepared for modeling.
```
cd example/
bash utils/unpack_dataset.sh
```
This will run two auxiliary python scripts to download the [MobiActV2 dataset](https://bmi.teicrete.gr/en/the-mobifall-and-mobiact-datasets-2/) from S3 and to transform the raw data and place it under `data/`.
