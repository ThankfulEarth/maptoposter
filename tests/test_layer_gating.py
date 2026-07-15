from create_map_poster import resolve_layers


def test_all_layers_default_on():
    assert resolve_layers(False, False, False, False, False) == {
        "water": True,
        "parks": True,
        "roads": True,
        "buildings": False,
        "labels": True,
    }


def test_disable_water_only():
    assert resolve_layers(True, False, False, False, False) == {
        "water": False,
        "parks": True,
        "roads": True,
        "buildings": False,
        "labels": True,
    }


def test_disable_all():
    assert resolve_layers(True, True, True, False, True) == {
        "water": False,
        "parks": False,
        "roads": False,
        "buildings": False,
        "labels": False,
    }


def test_buildings_enabled():
    assert resolve_layers(False, False, False, True, False) == {
        "water": True,
        "parks": True,
        "roads": True,
        "buildings": True,
        "labels": True,
    }


def test_labels_disabled():
    assert resolve_layers(False, False, False, False, True) == {
        "water": True,
        "parks": True,
        "roads": True,
        "buildings": False,
        "labels": False,
    }
