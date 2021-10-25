"""Tools for working with combinations of loads.

>>> from ceng.load Combination
>>> @Combination("1.6*D & 1.2*L & 0.5*(S | Lr | W)").function
... def my_combo(D, L, S, Lr, W): ...
...
>>> my_combo(1, 2, 3, 4, 5)
array([5.5, 6. , 6.5])

The result represents the 3 different answers from this load combination.
"""

import re
import ast
import dataclasses
from typing import TypeVar, Generic, Callable
import numpy as np
import numpy.typing as npt
import numba as nb
import wrapt

T = TypeVar('T')

_valid_operators = list("&|()")
_load_combination_re = re.compile("|".join("\\"+op for op in _valid_operators))
_multiply_re = re.compile(r"\*")


class LoadCombinationExpressionError(Exception):
    pass


def _get_identifiers(expr):
    """Parse a load combination string. Return a tuple of valid identifiers used in the string."""

    expr_lst = [s.strip() for s in _load_combination_re.split(expr) if s.strip()]
    identifiers = []

    for sub_expr in expr_lst:
        *lhs, rhs = (s.strip() for s in _multiply_re.split(sub_expr))
        # None is the case when the RHS is a parenthesized expression
        maybe_identifier = rhs if rhs else None
        for maybe_number in lhs:
            if not maybe_number.replace('.', '1').isdecimal():
                # LHS of multiplication operation has to be a number
                raise LoadCombinationExpressionError(expr, sub_expr)

        if maybe_identifier is not None:
            if not maybe_identifier.isidentifier():
                raise LoadCombinationExpressionError(expr, sub_expr)
            else:
                identifiers.append(maybe_identifier)

    try:
        ast.parse(expr)
    except Exception as e:
        raise LoadCombinationExpressionError(expr) from e

    return tuple(identifiers)


@dataclasses.dataclass
class _Factored(Generic[T]):
    """Represents a load type with a specified load factor.

    To be used in composing load expressions. E.g.:

    1.6*D & 1.2*L & 0.5*(S | Lr | W)

    Which means:

    1.6 dead load AND 1.2 live load AND 0.5 of either: snow load, live roof load, OR wind load
    """

    load_type: T
    factor: float = 1.0

    def __rmul__(self, other):
        return type(self)(self.load_type, self.factor * other)

    def __or__(self, other):
        if isinstance(other, _Factored):
            return _GroupOr((self, other))
        return NotImplemented

    def __ror__(self, other):
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, _Factored):
            return _GroupAnd((self, other))
        return NotImplemented

    def __rand__(self, other):
        return NotImplemented


class _Group(tuple[_Factored]):
    """_Factored objects that have been combined"""

    @property
    def matrix(self):
        factor_arr = np.array([factored.factor for factored in self])
        if isinstance(self, _GroupOr):
            return np.diag(factor_arr)
        elif isinstance(self, _GroupAnd):
            return factor_arr.reshape((1, factor_arr.shape[0]))


class _GroupOr(_Group):
    """_Factored objects that have been `__or__`ed together

        factored_load_type_a | factored_load_type_b
    """

    def __or__(self, other):
        if isinstance(other, _Factored):
            return type(self)((*self, other))
        return NotImplemented

    def __ror__(self, other):
        return NotImplemented

    def __and__(self, other):
        return NotImplemented

    def __rand__(self, other):
        if isinstance(other, _Factored):
            other = _GroupAnd((other,))
        if isinstance(other, (_GroupOr, _GroupAnd)):
            return tuple((other, self))
        if isinstance(other, tuple):
            return tuple((*other, self))
        return NotImplemented

    def __rmul__(self, other):
        return type(self)(type(v)(v.load_type, other*v.factor) for v in self)


class _GroupAnd(_Group):
    """_Factored objects that have been `__and__`ed together

        factored_load_type_a & factored_load_type_b
    """

    def __or__(self, other):
        return  NotImplemented

    def __ror__(self, other):
        return  NotImplemented

    def __and__(self, other):
        if isinstance(other, _Factored):
            return type(self)((*self, other))
        return NotImplemented

    def __rand__(self, other):
        return  NotImplemented


@dataclasses.dataclass(frozen=True)
class Combination:
    """A callable expression of the combination of multiple loads."""

    expr: str
    matrix: npt.ArrayLike = dataclasses.field(init=False, repr=False, compare=False)

    _identifiers: list[str] = dataclasses.field(init=False, repr=False)
    _call_handler: Callable[..., npt.ArrayLike] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_identifiers", _get_identifiers(self.expr))
        self._init_matrix()
        self._init_call_handler()

    def __str__(self):
        return self.expr

    def __and__(self, other):
        if isinstance(other, (_GroupAnd, _GroupOr)):
            return type(self)((*self, other))
        return  NotImplemented

    def _init_matrix(self):
        """Initialize a numpy array representing the load combination."""

        ns = {k:_Factored(k) for k in self._identifiers}
        try:
            expr_eval = eval(self.expr, ns)
        except Exception as e:
            raise LoadCombinationExpressionError(self.expr) from e

        if isinstance(expr_eval, _Factored):
            group_tup = (_GroupAnd((expr_eval,)),)
        elif isinstance(expr_eval, _GroupAnd):
            group_tup = (expr_eval,)
        elif isinstance(expr_eval, tuple):
            group_tup = expr_eval
        else:
            raise LoadCombinationExpressionError(f"{self.expr!r} evaluated to type {type(expr_eval).__qualname__}")

        arr_seq = [group.matrix for group in group_tup]
        object.__setattr__(self, "matrix", _row_by_row_concatenation_of_array_seq(arr_seq))

    def _init_call_handler(self):

        # TODO: create handler using AST

        comma_sep_identifiers = ", ".join(self._identifiers)
        f_src_template = f"""
@{{decorator}}
def {{func_name}}({comma_sep_identifiers}):
    {{func_body}}"""

        ns = dict(vectorize=nb.vectorize, njit=nb.njit)
        row_funcs = list()

        for factors in self.matrix:
            # vectorize each of the row funcs to support broadcasting
            vectorized_src_format_dict = dict(
                decorator = "vectorize",
                func_name = "vectorized",
                func_body = "return " + " + ".join(f"{f}*{i}" for f, i in zip(factors, self._identifiers))
            )
            vectorized_src = f_src_template.format(**vectorized_src_format_dict)

            # jit each of the row funcs to support kwd args
            njited_src_format_dict = dict(
                decorator = "njit",
                func_name = "njited",
                func_body = f"return vectorized({comma_sep_identifiers})"
            )
            njited_src = f_src_template.format(**njited_src_format_dict)

            src = vectorized_src + "\n\n\n" + njited_src

            ns_copy = ns.copy()
            exec(src, ns_copy)
            row_funcs.append(ns_copy["njited"])

        def _call_handler(*args, **kwargs):
            return np.array([f(*args, **kwargs) for f in row_funcs]).T

        # prime the function for the float type (this seems slightly faster than providing dtype info)
        _call_handler(*(1. for _ in range(len(self._identifiers))))

        object.__setattr__(self, "_call_handler", _call_handler)

    def _decorator(self, func, inner_dec=lambda f: f):

        @wrapt.decorator
        def combine_the_loads_wrapper(wrapped, instance, args, kwargs):
            return self(*args, **kwargs)

        return combine_the_loads_wrapper(inner_dec(func))

    def function(self, func):
        """Decorator to apply to a load combination function. Automatically implements load combination.

        Example:
        >>> @Combination("1.6*D & 1.2*L & 0.5*(S | Lr | W)").function
        ... def combo(D, L, S, Lr, W):
        ...    pass
        ...

        Produces:

        >>> combo(1, 2, 3, 4, 5)
        array([5.5, 6. , 6.5])
        """

        return self._decorator(func)

    def method(self, func):
        """Decorator to apply to a load combination method. Automatically implements load combination.

        Example:
        >>> class C:
        ...     @Combination("1.6*D & 1.2*L & 0.5*(S | Lr | W)").method
        ...     def combo(self, D, L, S, Lr, W):
        ...         pass
        ...

        Produces:

        >>> C().combo(1, 2, 3, 4, 5)
        array([5.5, 6. , 6.5])
        """

        return self._decorator(func)

    def staticmethod(self, func):
        """Decorator to apply to a load combination method. Automatically implements load combination.

        Example:
        >>> class C:
        ...     @Combination("1.6*D & 1.2*L & 0.5*(S | Lr | W)").staticmethod
        ...     def combo(D, L, S, Lr, W):
        ...         pass
        ...

        Produces:

        >>> C().combo(1, 2, 3, 4, 5)
        array([5.5, 6. , 6.5])
        """

        return self._decorator(func, staticmethod)

    def classmethod(self, func):
        """Decorator to apply to a load combination method. Automatically implements load combination.

        Example:
        >>> class C:
        ...     @Combination("1.6*D & 1.2*L & 0.5*(S | Lr | W)").classmethod
        ...     def combo(cls, D, L, S, Lr, W):
        ...         pass
        ...

        Produces:

        >>> C().combo(1, 2, 3, 4, 5)
        array([5.5, 6. , 6.5])
        """

        return self._decorator(func, classmethod)

    def __call__(self, *args, **kwargs):
        return self._call_handler(*args, **kwargs)


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
