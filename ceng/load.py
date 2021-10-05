import dataclasses
from functools import wraps, partial
from inspect import signature, ismethod
from typing import TypeVar, Generic
import numpy as np
import wrapt

T = TypeVar('T')


@dataclasses.dataclass
class Factored(Generic[T]):
    load_type: T
    factor: float = 1.0

    def __rmul__(self, other):
        return type(self)(self.load_type, self.factor * other)

    def __or__(self, other):
        if isinstance(other, Factored):
            return GroupOr((self, other))
        return NotImplemented

    def __ror__(self, other):
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, Factored):
            return GroupAnd((self, other))
        return NotImplemented

    def __rand__(self, other):
        return NotImplemented


class Group(tuple[Factored]):
    """Factored objects that have been combined"""

    @property
    def matrix(self):
        factor_arr = np.array([factored.factor for factored in self])
        if isinstance(self, GroupOr):
            return np.diag(factor_arr)
        elif isinstance(self, GroupAnd):
            return factor_arr.reshape((1, factor_arr.shape[0]))


class GroupOr(Group):
    """FactoredLoadType objects that have been `__or__`ed together

        factored_load_type_a | factored_load_type_b
    """

    def __or__(self, other):
        if isinstance(other, Factored):
            return type(self)((*self, other))
        return NotImplemented

    def __ror__(self, other):
        return NotImplemented

    def __and__(self, other):
        return NotImplemented

    def __rand__(self, other):
        if isinstance(other, Factored):
            other = GroupAnd((other,))
        if isinstance(other, (GroupOr, GroupAnd)):
            return Combination((other, self))
        return NotImplemented

    def __rmul__(self, other):
        return type(self)(type(v)(v.load_type, other*v.factor) for v in self)


class GroupAnd(Group):
    """FactoredLoadType objects that have been `__and__`ed together

        factored_load_type_a & factored_load_type_b
    """

    def __or__(self, other):
        return  NotImplemented

    def __ror__(self, other):
        return  NotImplemented

    def __and__(self, other):
        if isinstance(other, Factored):
            return type(self)((*self, other))
        return NotImplemented

    def __rand__(self, other):
        return  NotImplemented


class Combination(tuple[Group]):

    def __and__(self, other):
        if isinstance(other, (GroupAnd, GroupOr)):
            return type(self)((*self, other))
        return  NotImplemented


def _get_equations_matrix(load_factor_groups):
    arr_seq = [group.matrix for group in load_factor_groups]
    return _row_by_row_concatenation_of_array_seq(arr_seq)


def load_combination(*dec_args):
    *dec_args, func = dec_args if any(callable(v) for v in dec_args[-1:]) else (*dec_args, None)

    if func is None:
        return partial(load_combination, *dec_args)
    else:
        load_expr, = dec_args

    if isinstance(load_expr, Factored):
        load_combination_obj = Combination((GroupAnd((load_expr,)),))
    elif isinstance(load_expr, GroupAnd):
        load_combination_obj = Combination((load_expr,))
    else:
        load_combination_obj = load_expr

    mat = _get_equations_matrix(load_combination_obj)
    sig = signature(func)

    @wrapt.decorator
    def combine_the_loads_wrapper(wrapped, instance, args, kwargs):
        is_a_method = ismethod(wrapped)
        bound = sig.bind(instance, *args, **kwargs) if is_a_method else sig.bind(*args, **kwargs)
        # mat is A, input is B. do A X B.T, and transpose result
        return (mat @ np.array(list(bound.arguments.values())[is_a_method:]).T).T
    return combine_the_loads_wrapper(func)


def _row_by_row_concatenation_of_array_seq(arr_seq):
    """Combine two arrays using row by row concatenation.

    Example
    =======

    arr0:           arr1:
    [[1,2,3],       [[1,2],
    [4,5,6],        [3,4]]
    [7,8,9]]

    Into form of:

    [[1,2,3,1,2],
    [1,2,3,3,4],
    [4,5,6,1,2],
    [4,5,6,3,4],
    [7,8,9,1,2],
    [7,8,9,3,4]]
    """

    seq_rows = np.array([arr.shape[0] for arr in arr_seq])
    products = np.array([seq_rows[i:].prod() for i,_ in enumerate(seq_rows)])
    rows = seq_rows.prod()

    return np.hstack([
        np.tile(np.repeat(arr, repeats=prod//arr_rows, axis=0), (rows//prod ,1))
        for arr, arr_rows, prod in zip(arr_seq, seq_rows, products)
    ])
