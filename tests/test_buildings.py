from create_map_poster import buildings_allowed


def test_small_bbox_allowed():
    assert buildings_allowed(50.0) is True


def test_at_cap_allowed():
    assert buildings_allowed(200.0) is True


def test_over_cap_rejected():
    assert buildings_allowed(200.1) is False


def test_custom_cap():
    assert buildings_allowed(120.0, cap_km2=100.0) is False
