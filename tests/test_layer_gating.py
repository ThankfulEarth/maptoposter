from create_map_poster import resolve_layers


def test_all_layers_default_on():
    assert resolve_layers(False, False, False) == {"water": True, "parks": True, "roads": True}


def test_disable_water_only():
    assert resolve_layers(True, False, False) == {"water": False, "parks": True, "roads": True}


def test_disable_all():
    assert resolve_layers(True, True, True) == {"water": False, "parks": False, "roads": False}
