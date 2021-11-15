from collections.abc import Mapping


class IterableDict(Mapping):
    """A dict that is constructed using iterables of keys rather than individual keys.

    >>> i_dict = IterableDict((((1,2,3), "spam"), ((4,5,6), "eggs")))
    >>> i_dict
    IterableDict({(1, 2, 3): 'spam', (4, 5, 6): 'eggs'})
    >>> i_dict[1]
    'spam'
    >>> i_dict[4]
    'eggs'
    """
    root: Mapping
    _dict: dict

    def __init__(self, args):
        self.root = dict(args)
        self._dict = {k: v for keys, v in self.root.items() for k in keys}

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        yield from self._dict.keys()

    def __getitem__(self, k):
        return self._dict.__getitem__(k)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.root.__repr__()})"
