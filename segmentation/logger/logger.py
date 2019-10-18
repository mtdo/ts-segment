import torch


class Logger(object):
    """ A Logger object is used to log and print metrics during training.
    
    Args:
        evaluator: Ignite Engine used for model evaluation.
        train_loader: PyTorch DataLoader object for the training set.
        validation_loader: PyTorch DataLoader object for the validation set.
    """
    def __init__(self, evaluator, train_loader, validation_loader):
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.metrics = {"training": [], "validation": []}

    def __call__(self, engine):
        print(f"Epoch {engine.state.epoch} training results")
        print("-------------------------")

        # Training
        self.evaluator.run(self.train_loader)
        self.metrics["training"].append(self.evaluator.state.metrics)
        print(
            f"Loss: {round(self.metrics['training'][-1]['loss'], 3)},\
              Samplewise accuracy: {round(self.metrics['training'][-1]['samplewise_accuracy'], 3)},\
              Mean IoU: {round(self.metrics['training'][-1]['mean_iou'], 3)},\
              Frequency weighted IoU: {round(self.metrics['training'][-1]['frequency_weighted_iou'], 3)}"
        )
        print()

        # Validation
        self.evaluator.run(self.validation_loader)
        self.metrics["validation"].append(self.evaluator.state.metrics)
