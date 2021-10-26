from collections import defaultdict
from ceng.common import InfoArray, iter_arg_attrs_if_attr_exists
from scipy.interpolate import interp1d
import numpy as np

# Note: the scipy interp2d algorithm doesn't give the right results for twice interpolation of figures with multiple
# curves. no idea why, so have to write own interp2d algorithm.


def _interp1d_rows(inner_arr, col_arr, bounds_error, fill_value, dependent_2d):
    if dependent_2d==True:
        # z array is 2d so rows will be dependent argument to interp1d
        return [interp1d(col_arr, row, bounds_error=bounds_error, fill_value=fill_value) for row in inner_arr]
    # x or y is 2d so rows will be independent argument to interp1d
    return [interp1d(row, col_arr, bounds_error=bounds_error, fill_value=fill_value) for row in inner_arr]


def _get_row_1darr_col_1darr_and_inner_2darr(x_seq, y_seq, z_seq, first_dependent_axis):
    x_arr, y_arr, z_arr = (np.array(seq) for seq in (x_seq, y_seq, z_seq))
    x_d, y_d, z_d = (len(arr.shape) for arr in (x_arr, y_arr, z_arr))

    if not any(d==2 for d in (x_d, y_d, z_d)):
        raise ValueError("a 2d sequence or array is required")

    if all(d==1 for d in (x_d, y_d)):
        # z assumed 2d, first dependent axis is x
        row_arr, col_arr = (x_arr, y_arr) if first_dependent_axis == 0 else (y_arr, x_arr)
        inner_arr = InfoArray(z_arr, info="z")
    elif all(d==1 for d in (x_d, z_d)):
        # y assumed 2d, first dependent axis is x
        y_arr_maybe_transposed = y_arr.transpose() if first_dependent_axis == 1 else y_arr
        row_arr, col_arr = (x_arr, z_arr) if first_dependent_axis == 0 else (z_arr, x_arr)
        inner_arr = InfoArray(y_arr_maybe_transposed, info="y")
    elif all(d==1 for d in (y_d, z_d)):
        # x assumed 2d, first dependent axis is y
        x_arr_maybe_transposed = x_arr.transpose() if first_dependent_axis == 1 else x_arr
        row_arr, col_arr = (y_arr, z_arr) if first_dependent_axis == 0 else (z_arr, y_arr)
        inner_arr = InfoArray(x_arr_maybe_transposed, info="x")
    else:
        raise ValueError("two 1d sequences or arrays are required")

    if (row_arr.size, col_arr.size) != inner_arr.shape:
        raise ValueError(f"Shape of inner 2d array does not match sizes of outer arrays with first dependent at axis "
                         f"{first_dependent_axis}")

    return row_arr, col_arr, inner_arr


def _twice_interpolate_2d_using_list_of_interpolation_functions(col_v, row_v, row_arr, interp1d_rows,
                                                                bounds_error, fill_value):
    """Interpolate the function.

     Parameters
     ----------
      col_v, row_v: array_like
         column and row values to be interpolated.
     row_arr : list of interpolation functions
         values to used to create the interpolation function for second interpolation step
     interp1d_rows : list of interpolation functions
         1d interpolation functions to be used for first interpolation step

     Returns
     -------
     The interpolated value(s).
    """

    temp_x = [f(col_v) for f in interp1d_rows]
    return interp1d(row_arr, temp_x, bounds_error=bounds_error, fill_value=fill_value)(row_v)


def _twice_interp1d(x_seq, y_seq, z_seq, first_dependent_axis, bounds_error, fill_value):

    row_arr, col_arr, inner_arr = _get_row_1darr_col_1darr_and_inner_2darr(x_seq, y_seq, z_seq, first_dependent_axis)
    dependent_2d = True if inner_arr.info=="z" else False
    interp1d_rows = _interp1d_rows(inner_arr, col_arr, bounds_error, fill_value, dependent_2d=dependent_2d)

    def interpolate(x, y):

        if dependent_2d:
            axis_dict = {first_dependent_axis: x, first_dependent_axis^1: y}
            row_v, col_v = (axis_dict[i] for i in (0,1))
        else:
            row_v, col_v = y, x
        return _twice_interpolate_2d_using_list_of_interpolation_functions(col_v, row_v, row_arr, interp1d_rows,
                                                                           bounds_error, fill_value)
    return interpolate


def interp1d_twice(x, y, z, axis=0, bounds_error=True, fill_value=None):
    """Interpolate twice from many curves.

    x, y and z are values used to approximate some function f: z = f(x, y) which returns an interpolated value.

    Parameters
    ----------
    x,y : array_like of numbers
        the independent data of the curve; either both 1d, or one of x and y can be 2d if z is 1d
    z : array_like of numbers
        the dependent data of the curve; either 2d or if one of x or y are 2d, then 1d
    axis: int (0 or 1)
        axis of 2d argument assumed to correspond to the first dependent axis:
            - first dependent axis is x if either y or z is 2d
            - first dependent axis is y if x is 2d
    bounds_error : bool, optional
        if True, when interpolated values are requested outside the domain of the input data (x,y),
        a ValueError is raised. If False, then fill_value is used.
    fill_value : number, optional
        If provided, the value to use for points outside the interpolation domain. If omitted (None),
        values outside the  domain are extrapolated via nearest-neighbor extrapolation.

    Returns
    -------
    interpolator function (interpolant)
        a function that interpolates values

    Raises
    ------
    ValueError
        when unexpected or incompatible array shapes are provided
    """

    return _twice_interp1d(x, y, z, first_dependent_axis=axis, bounds_error=bounds_error, fill_value=fill_value)


def _keys_view_for_all_args_if_arg_has_keys(*args):
    """A single view of the keys of all the arguments that have them. Arguments are ignored if they don't have keys.

    Returns None if there is not at least one mapping. Raises ValueError if there are conflicting sets of keys in any
    mappings."""

    keys_iterator = (keys() for keys in iter_arg_attrs_if_attr_exists(*args, attr="keys"))
    for keys_view in keys_iterator:
        if not all(kv == keys_view for kv in keys_iterator):
            raise ValueError("mappings must have the same keys")
        break
    else:
        return None
    return keys_view


def interp_dict(x, y, z=None, axis=0, bounds_error=True, fill_value=None):
    """Produce a dictionary of interpolation functions ("interpolants").

    One or more of x,y,z must be a mapping and any mappings must have identical sets of keys.

    If z in omitted, interpolation is 1d. Otherwise it is twice 1d and exactly one of x, y, z must be 2d."""

    # get common keys view first (checks for consistency)
    keys_view = _keys_view_for_all_args_if_arg_has_keys(x, y, z)

    # turn x y z into mappings (if they aren't already)
    x_dct, y_dct, z_dct = (input if hasattr(input, "keys") else defaultdict(f)
                           for input, f in zip((x, y, z),
                                               (lambda: x,
                                                lambda: y,
                                                lambda: z)))

    # suss out the independent and dependents variables, and the interpolation method
    dependent_dct = y_dct if z is None else z_dct
    x_and_maybe_y_dicts = (x_dct,) if z is None else (x_dct, y_dct)
    interp_func = interp1d if z is None else interp1d_twice

    kwargs = dict(axis=axis, bounds_error=bounds_error, fill_value=fill_value)
    if z is None:
        if axis != 0:
            raise ValueError("axis is not used for 1d interpolation")
        del kwargs["axis"]

    result = {}
    for key in keys_view:
        x_and_maybe_y = (x_and_maybe_y_dict[key] for x_and_maybe_y_dict in x_and_maybe_y_dicts)
        result[key] = interp_func(*x_and_maybe_y, dependent_dct[key], **kwargs)

    return result
