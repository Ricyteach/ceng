from inspect import signature
from ceng.common import flatten

import numpy as np
import pytest
from ceng.load import load_combination, Factored


@pytest.fixture
def D():
    return Factored("D")


@pytest.fixture
def L():
    return Factored("L")


@pytest.fixture
def Lr():
    return Factored("Lr")


@pytest.fixture
def S():
    return Factored("S")


@pytest.fixture
def R():
    return Factored("R")


@pytest.fixture
def W():
    return Factored("W")


combo_strs = [
    "(1.4 * D)",
    "(1.2 * D & 1.6 * L & 0.5 * (Lr | S | R))",
    "(1.2 * D & 1.6 * (S | Lr | R) & (1.0 * L | 0.5 * W))",
    "(1.2 * D & W & L & 0.5 * (Lr | S | R))",
    "(0.9 * D & W)",
    "(D)",
    "(D & L)",
    "(D & (S | Lr | R))",
    "(D & 0.75 * L & 0.75 * (Lr | S | R))",
    "(D & 0.6 * W)",
    "(D & 0.75 * L & 0.75 * 0.6 * W & 0.75 * (Lr | S | R))",
    "(0.6 * D & 0.6 * W)",
]


@pytest.fixture(params=range(len(combo_strs)), ids=combo_strs)
def combo_list_idx(request):
    return request.param


@pytest.fixture
def combo(combo_list_idx):
    return combo_strs[combo_list_idx]


def test_factored_objects(combo):
    assert combo


@pytest.fixture
def combo_obj(combo, D, L, Lr, S, R, W):
    return eval(combo)


def test_combo_obj(combo_obj):
    assert combo_obj


@pytest.fixture
def load_combination_function_factory(combo_obj):
    return load_combination(combo_obj)


def test_load_combination_function_creation(load_combination_function_factory):
    assert load_combination_function_factory


@pytest.fixture
def load_combination_func(load_combination_function_factory, combo_obj):
    args = flatten(combo_obj)
    func = eval(f"lambda {', '.join(a.load_type for a in args)}: None")
    return load_combination_function_factory(func)


combo_max_func_expected = [
    lambda D, L, Lr, S, R, W: (1.4 * D),
    lambda D, L, Lr, S, R, W: (1.2 * D + 1.6 * L + 0.5 * max(Lr, S, R)),
    lambda D, L, Lr, S, R, W: (1.2 * D + 1.6 * max(S, Lr, R) + max(1.0 * L, 0.5 * W)),
    lambda D, L, Lr, S, R, W: (1.2 * D + W + L + 0.5 * max(Lr, S, R)),
    lambda D, L, Lr, S, R, W: (0.9 * D + W),
    lambda D, L, Lr, S, R, W: (D),
    lambda D, L, Lr, S, R, W: (D + L),
    lambda D, L, Lr, S, R, W: (D + max(S, Lr, R)),
    lambda D, L, Lr, S, R, W: (D + 0.75 * L + 0.75 * max(Lr, S, R)),
    lambda D, L, Lr, S, R, W: (D + 0.6 * W),
    lambda D, L, Lr, S, R, W: (D + 0.75 * L + 0.75 * 0.6 * W + 0.75 * max(Lr, S, R)),
    lambda D, L, Lr, S, R, W: (0.6 * D + 0.6 * W),
]


@pytest.fixture(params=[
    {"D": 1, "L": 1, "Lr": 1, "S": 1, "R": 1, "W": 1},
    {"D": 1, "L": 2, "Lr": 3, "S": 4, "R": 5, "W": 6},
    {"D": -6, "L": -5, "Lr": -4, "S": -3, "R": -2, "W": -1},
])
def load_combo_input(request):
    return request.param


@pytest.fixture
def load_combo_result_expected(load_combo_input, combo_list_idx):
    return combo_max_func_expected[combo_list_idx](*load_combo_input.values())


def test_combo_max_result(load_combo_input, load_combination_func, load_combo_result_expected):
    s = signature(load_combination_func)
    result = load_combination_func(**{k:v for k,v in load_combo_input.items() if k in s.parameters})
    assert np.max(result) == load_combo_result_expected


def test_load_combo_as_method(D, L):
    class C:
        @load_combination(D & L)
        def m(self, D, L): ...

    assert np.max(C().m(1, 1)) == 2
