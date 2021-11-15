from ceng.lookup import IterableDict
import pytest


@pytest.fixture(name="iterable_dict_args")
def iterable_dict_args_fixture(request):
    return request.param


@pytest.fixture(name="iterable_dict")
def iterable_dict_fixture(iterable_dict_args):
    return IterableDict(*iterable_dict_args)


some_keys_and_values = [
    [(range(0, 7), "spam"), (range(7, 27), "eggs")],
    [((0,), "spam"), ((7, 26), "eggs")]
]

parametrize_some_iterables = pytest.mark.parametrize("iterable_dict_args", some_keys_and_values, indirect=True)


@pytest.mark.parametrize("key, value", [
    (0, "spam"), (7, "eggs"), (26, "eggs")
])
@parametrize_some_iterables
def test_iterable_dict_getitem(iterable_dict, iterable_dict_args, key, value):
    """Test getting API"""
    assert iterable_dict[key] == value
    assert iterable_dict.get(key) == value
    keys, values = zip(iterable_dict_args)
    assert (iterable_dict.key_iterables, iterable_dict.values_of_iterables) == (keys, values)


@parametrize_some_iterables
def test_iterable_dict_api(iterable_dict):
    """Test get API and test immutable"""
    assert iterable_dict.get(-1, None) is None
    with pytest.raises(TypeError):
        iterable_dict["spam"] = None
