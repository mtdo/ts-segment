import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorFCN(nn.Module):
    """ A fully convolutional network for segmentation of motion sensor data from
    smartphones or wearable sensors.
    
    Args:
        n_input_channels (int): The number of input channels for the network. For example,
            if the input is the data of a triaxial accelerometer the number of input channels is 3.
        n_classes (int): The number of target classes for the model predictions.
        input_kernel_size (int): The number of samples to be used in the receptive field.
        n_filters (int): The filter parameter of the SensorFCN network architecture. Larger filter 
            parameter values result in larger and more powerful models.
        smoothing_kernel_size (int): The kernel size for the smoothing component of the SensorFCN
            architecture. Including a smoothing component generally results in better models,
            especially for the segmentation of sporadically occurring patterns.
    """
    def __init__(
        self,
        n_input_channels,
        n_classes,
        input_kernel_size,
        n_filters=16,
        smoothing_kernel_size=0,
    ):
        super(SensorFCN, self).__init__()
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes
        self.input_kernel_size = input_kernel_size
        self.n_filters = n_filters
        self.smoothing_kernel_size = smoothing_kernel_size

        # Encoding layers
        self.conv1 = nn.Conv1d(n_input_channels, n_filters, input_kernel_size)
        self.conv2 = nn.Conv1d(n_filters, 2 * n_filters, 5)
        self.conv3 = nn.Conv1d(2 * n_filters, 4 * n_filters, 3)

        # Decoding layer
        self.tconv1 = nn.ConvTranspose1d(
            4 * n_filters, n_filters, 2 + 4 + input_kernel_size
        )

        # Scoring layer
        self.score_conv = nn.Conv1d(n_filters, n_classes, 1, 1, 0)

        # Post scoring layer
        if smoothing_kernel_size > 0:
            self.smoothing_conv = nn.Conv1d(
                n_classes,
                n_classes,
                smoothing_kernel_size,
                1,
                smoothing_kernel_size // 2,
            )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.tconv1(x)
        x = self.score_conv(x)
        if self.smoothing_kernel_size > 0:
            x = self.smoothing_conv(x)
        return x
