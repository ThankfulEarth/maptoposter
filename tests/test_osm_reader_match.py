from osm_reader import OsmTagHandler


class FakeTags:
    def __init__(self, d): self._d = d
    def __contains__(self, k): return k in self._d
    def __getitem__(self, k): return self._d[k]


def test_true_matches_any_value():
    h = OsmTagHandler({"building": True})
    assert h._matches_tags(FakeTags({"building": "yes"})) is True
    assert h._matches_tags(FakeTags({"building": "house"})) is True
    assert h._matches_tags(FakeTags({"highway": "primary"})) is False


def test_scalar_still_exact():
    h = OsmTagHandler({"natural": "water"})
    assert h._matches_tags(FakeTags({"natural": "water"})) is True
    assert h._matches_tags(FakeTags({"natural": "wood"})) is False
