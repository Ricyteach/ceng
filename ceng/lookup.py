from collections.abc import Mapping


class IterableDict(Mapping):
    """A dict that is constructed using iterables of keys rather than individual keys.

    >>> IterableDict(((1,2,3), "spam"), ((4,5,6), "eggs"))
    {(1,2,3): "spam", (4,5,6): "eggs"}
    """
    key_iterables: tuple
    values_of_iterables: tuple
    _dict: dict
    _len: int

    def __init__(self, *args):
        self.key_iterables, self.values_of_iterables = zip(args)
        self._dict = {k: v for keys, v in args for k in keys}
        self._len = len(self._dict)

    def __len__(self):
        return self._len

    def __iter__(self):
        yield from self._dict.keys()

    def __getitem__(self, k):
        return self._dict.__getitem__(k)

    def __repr__(self):
        ...  # TODO
