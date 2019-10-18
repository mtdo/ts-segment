import torch
from ignite.engine.engine import Engine


def create_trainer(model, optimizer, loss_fn, device):
    """ Creates an ignite Engine instance for model training.
    
    Args:
        model: PyTorch model instance to be trained.
        optimizer: PyTorch optimizer to be used for model training.
        loss_fn: PyTorch loss function used for the model training.
        device: A string representing the used device (cpu or gpu).
    """
    model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        # Prepare data
        inputs, labels = batch
        inputs = inputs.permute(0, 2, 1).to(device, dtype=torch.float)
        labels = labels["y"].to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_evaluator(model, device, metrics={}):
    """ Creates an ignite Engine instance for model evaluation.
    
    Args:
        model: PyTorch model instance to be evaluated.
        device: A string representing the used device (cpu or gpu).
        metrics: A dictionary of the evaluation metrics.
    """
    model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # Prepare data
            inputs, labels = batch
            inputs = inputs.permute(0, 2, 1).to(device, dtype=torch.float)
            labels = labels["y"].to(device)

            outputs = model(inputs)
            return outputs, labels

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
