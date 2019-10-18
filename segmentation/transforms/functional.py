def xflip(frame):
    """ Flip the given sensor frame along the x-axis.
    
    Args:
        frame (torch.Tensor): Sensor frame to be flipped.
    """
    frame[1, :] = -frame[1, :]
    frame[2, :] = -frame[2, :]
    return frame


def yflip(frame):
    """ Flip the given sensor frame along the y-axis.
    
    Args:
        frame (torch.Tensor): Sensor frame to be flipped.
    """
    frame[0, :] = -frame[0, :]
    frame[2, :] = -frame[2, :]
    return frame


def zflip(frame):
    """ Flip the given sensor frame along the z-axis.
    
    Args:
        frame (torch.Tensor): Sensor frame to be flipped.
    """
    frame[0, :] = -frame[0, :]
    frame[1, :] = -frame[1, :]
    return frame
