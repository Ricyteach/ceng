from ceng.lookup import IterableDict
import pytest


@pytest.fixture(name="dict_args")
def iterable_dict_arg_fixture(request):
    return request.param


@pytest.fixture(name="iterable_dict_from_kv_pairs")
def iterable_dict_from_kv_pairs_fixture(dict_args):
    return IterableDict(dict_args)


@pytest.fixture(name="iterable_dict_from_root_dict")
def iterable_dict_from_root_dict_fixture(regular_dict):
    return IterableDict(regular_dict)


@pytest.fixture(name="regular_dict")
def regular_dict_fixture(dict_args):
    return dict(dict_args)


some_keys_and_values = [
    [(range(0, 7), "spam"), (range(7, 27), "eggs")],
    [((0,), "spam"), ((7, 26), "eggs")]
]

parametrize_some_iterables = pytest.mark.parametrize("dict_args", some_keys_and_values, indirect=True)


@pytest.mark.parametrize("key, value", [
    (0, "spam"), (7, "eggs"), (26, "eggs")
])
@parametrize_some_iterables
def test_iterable_dict_getitem(iterable_dict_from_kv_pairs, iterable_dict_from_root_dict, key, value):
    """Test the getting API"""
    assert iterable_dict_from_kv_pairs[key] == value
    assert iterable_dict_from_kv_pairs.get(key) == value
    assert iterable_dict_from_root_dict[key] == value
    assert iterable_dict_from_root_dict.get(key) == value


@parametrize_some_iterables
def test_iterable_dict_api(iterable_dict_from_kv_pairs, iterable_dict_from_root_dict):
    """Test get with missing key API and test immutable"""
    assert iterable_dict_from_kv_pairs.get(-1, None) is None
    with pytest.raises(TypeError):
        iterable_dict_from_kv_pairs["spam"] = None
    assert iterable_dict_from_root_dict.get(-1, None) is None
    with pytest.raises(TypeError):
        iterable_dict_from_root_dict["spam"] = None


@parametrize_some_iterables
def test_root_dict(regular_dict, iterable_dict_from_kv_pairs, iterable_dict_from_root_dict):
    """Confirm part of API that creates/saves the root dictionary."""
    assert iterable_dict_from_kv_pairs.root == regular_dict
    assert iterable_dict_from_root_dict.root == regular_dict


def test_error_on_unhashable_iterable():
    """Be sure to mirror behavior of vanilla dict when unhashable iterables provided for keys"""
    with pytest.raises(TypeError):
        IterableDict((([1], "spam"),))


def test_error_on_incorrectly_provided_args():
    """Be sure to mirror behavior of vanilla dict when incorrect arguments provided"""
    with pytest.raises(ValueError):
        # mistakenly provide a two-tuple instead a tuple of two-tuples
        IterableDict(([1], "spam"))
