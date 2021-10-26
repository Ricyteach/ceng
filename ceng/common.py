import numpy as np


def flatten(maybe_seq):
    try:
        n = len(maybe_seq)
    except TypeError:
        return [maybe_seq]
    if n == 0:
        return maybe_seq
    elif n == 1:
        item = maybe_seq[0]
        try:
            return flatten(list(item))
        except TypeError:
            return [item]
    else:
        k = n // 2
        return flatten(maybe_seq[:k]) + flatten(maybe_seq[k:])


class InfoArray(np.ndarray):
    """Same as numpy array but can have an info attribute"""

    def __new__(cls, input_array, info=None):
        # Input array is an array_like
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)


def iter_arg_attrs_if_attr_exists(*args, attr: str):
    """Iterate over the specified attribute of the arguments if they exist."""

    yield from (a for input in args if (a := getattr(input, attr, None)) is not None)
