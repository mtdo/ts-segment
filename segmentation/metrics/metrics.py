import torch
from ignite.metrics.metric import Metric


class SamplewiseAccuracy(Metric):
    """ Segmentation samplewise accuracy. This metric can be attached to 
    an ignite evaluator engine and will return the samplewise accuracy
    for each epoch."""

    def reset(self):
        """ Resets the number of correctly predicted and total samples
        at the start of each epoch. """
        self._correct_samples = 0
        self._total_samples = 0

    def update(self, data):
        # Unpack data, assert shapes and get predictions
        outputs, labels = data
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update numbers of correctly predicted and total samples
        self._correct_samples += (outputs == labels).sum(dtype=torch.float)
        self._total_samples += torch.numel(outputs)

    def compute(self):
        return self._correct_samples / self._total_samples


class MeanAccuracy(Metric):
    """ Segmentation mean class accuracy. This metric can be attached to 
    an ignite evaluator engine and will return the mean class accuracy
    for each epoch."""

    def reset(self):
        """ Resets the classwise number of correctly predicted and total samples 
        at the start of each epoch. """
        self._correct_class_samples = {}
        self._total_class_samples = {}

    def update(self, data):
        # Unpack data, assert shapes and get predictions
        outputs, labels = data
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update correctly predicted and total precitions for each class in batch
        for label in torch.unique(labels):
            if not label in self._total_class_samples:
                self._correct_class_samples[label] = 0
                self._total_class_samples[label] = 0

            # Samples belonging to current class
            class_samples = labels == label

            # Correctly predicted samples and total samples for current class in batch
            correct_samples = (outputs[class_samples] == label).sum(dtype=torch.float)
            total_samples = class_samples.sum(dtype=torch.float)
            self._correct_class_samples[label] += correct_samples
            self._total_class_samples[label] += total_samples

    def compute(self):
        accuracies = []
        for label in self._total_class_samples:
            correct_samples = self._correct_class_samples[label]
            total_samples = self._total_class_samples[label]
            accuracies.append(correct_samples / total_samples)
        return torch.mean(torch.tensor(accuracies))


class MeanIoU(Metric):
    """ Segmentation mean class IoU. This metric can be attached to 
    an ignite evaluator engine and will return the mean IoU for each epoch."""

    def reset(self):
        """ Resets the classwise intersection and union at the start of each epoch."""
        self._class_intersection = {}
        self._class_union = {}

    def update(self, data):
        # Unpack data, assert shapes and get predictions
        outputs, labels = data
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update intersection and union for each class in batch
        for label in torch.unique(labels):
            if not label in self._class_intersection:
                self._class_intersection[label] = 0
                self._class_union[label] = 0

            # Intersection and union of current class
            intersection = (
                ((labels == label) & (outputs == label)).sum(dtype=torch.float).item()
            )
            union = (
                ((labels == label) | (outputs == label)).sum(dtype=torch.float).item()
            )
            self._class_intersection[label] += intersection
            self._class_union[label] += union

    def compute(self):
        ious = []
        for label in self._class_intersection:
            total_intersection = self._class_intersection[label]
            total_union = self._class_union[label]
            ious.append(total_intersection / total_union)
        return torch.mean(torch.tensor(ious))


class FrequencyWeightedIoU(Metric):
    """ Segmentation frequency weighted class IoU. This metric can be attached to 
    an ignite evaluator engine and will return the frequency weighted IoU for each epoch."""

    def reset(self):
        """ Resets the classwise intersection, union, class samples and total samples at the start of each epoch."""
        self._class_intersection = {}
        self._class_union = {}
        self._class_samples = {}
        self._total_samples = 0

    def update(self, data):
        # Unpack data, assert shapes and get predictions
        outputs, labels = data
        assert outputs.shape[0] == labels.shape[0]
        outputs = outputs.argmax(1)

        # Update intersection, union, class and total samples
        for label in torch.unique(labels):
            if not label in self._class_intersection:
                self._class_intersection[label] = 0
                self._class_union[label] = 0
                self._class_samples[label] = 0

            # Samples belonging to current class
            class_samples = labels == label

            # Total samples, class samples, and intersection and union of current class
            self._total_samples += class_samples.sum(dtype=torch.float).item()
            self._class_samples[label] += class_samples.sum(dtype=torch.float).item()
            intersection = (
                ((labels == label) & (outputs == label)).sum(dtype=torch.float).item()
            )
            union = (
                ((labels == label) | (outputs == label)).sum(dtype=torch.float).item()
            )
            self._class_intersection[label] += intersection
            self._class_union[label] += union

    def compute(self):
        ious = []
        for label in self._class_intersection:
            total_samples = self._total_samples
            class_samples = self._class_samples[label]
            class_intersection = self._class_intersection[label]
            class_union = self._class_union[label]
            ious.append(class_samples * class_intersection / class_union)
        return torch.tensor(ious).sum().item() / total_samples
