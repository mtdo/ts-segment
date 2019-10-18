from . import functional as F
import random


class Compose(object):
    """ Composes multiple transforms together.
    
    Args:
        transforms: Transformation objects to be composed together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame):
        for t in self.transforms:
            frame = t(frame)
        return frame

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "     {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomXFlip(object):
    """ Flip the given sensor frame along the x-axis randomly with a given probability.
    
    Args:
        p (float): probability of the frame being flipped.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, frame):
        if random.random() < self.p:
            return F.xflip(frame)
        return frame

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomYFlip(object):
    """ Flip the given sensor frame along the y-axis randomly with a given probability.
    
    Args:
        p (float): probability of the frame being flipped.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, frame):
        if random.random() < self.p:
            return F.yflip(frame)
        return frame

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomZFlip(object):
    """ Flip the given sensor frame along the z-axis randomly with a given probability.
    
    Args:
        p (float): probability of the frame being flipped.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, frame):
        if random.random() < self.p:
            return F.zflip(frame)
        return frame

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)
