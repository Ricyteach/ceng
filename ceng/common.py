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
