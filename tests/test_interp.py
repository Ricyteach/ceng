import pytest
import numpy as np
from ceng.interp import interp1d_twice


@pytest.mark.parametrize("x, y, z", [
    (1.5, 1.5, 3.0),
    (1, 1.5, 1.5),
    (1.5, 1, 2.5),
])
def test_twice_interp1d_with_2d_z(x, y, z):
    f = interp1d_twice(np.array([1, 2]), np.array([1, 2, 3]), np.array([[1, 2, 3],
                                                                        [4, 5, 6]]),
                       axis=0, bounds_error=True, fill_value=None)
    assert f(x, y) == z


@pytest.mark.skip(reason="YAGNI?")
@pytest.mark.parametrize("x, y, z", [
    (1.0, 1.0, 1.0),
    (3.0, 2.0, 1.0),
    (5.0, 3.0, 1.0),
    (1.5, 1.5, 1.0),
])
def test_twice_interp1d_with_2d_x(x, y, z):
    """This is a test that might be required to pass in the future if ever a chart/table comes up demanding it.
    I hope not because it will be a real PITA.
    See:
    https://github.com/scipy/scipy/issues/14735
    """
    f = interp1d_twice(np.array([[1, 2],
                                 [3, 4],
                                 [5, 6]]), np.array([1, 2, 3]), np.array([1, 2]),
                       axis=0, bounds_error=False, fill_value=None)
    assert f(x, y) == z
