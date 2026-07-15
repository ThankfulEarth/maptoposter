import geopandas as gpd
from shapely.geometry import Point

from create_map_poster import select_labels


def _gdf(rows):
    return gpd.GeoDataFrame(rows, geometry=[r["geometry"] for r in rows], crs="EPSG:4326")


def test_ranks_and_caps():
    rows = [
        {"name": "Big City", "place": "city", "geometry": Point(10.0, 55.0)},
        {"name": "Small Village", "place": "village", "geometry": Point(10.1, 55.1)},
        {"name": "A Town", "place": "town", "geometry": Point(10.2, 55.2)},
    ]
    out = select_labels(_gdf(rows), max_labels=2, min_dist_deg=0.001)
    assert len(out) == 2
    # city outranks town outranks village
    assert "Big City" in list(out["name"])
    assert "Small Village" not in list(out["name"])


def test_dedup_by_min_distance():
    rows = [
        {"name": "One", "place": "city", "geometry": Point(10.0, 55.0)},
        {"name": "Two", "place": "city", "geometry": Point(10.00001, 55.00001)},
    ]
    out = select_labels(_gdf(rows), max_labels=10, min_dist_deg=0.01)
    assert len(out) == 1  # second dropped as too close
