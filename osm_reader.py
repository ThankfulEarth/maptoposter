"""
OSM Offline Reader - Parse PBF files into GeoDataFrames.

Replaces OSMnx for offline mode, using cached PBF extracts.
Includes a second-level cache for processed GeoDataFrames.
"""

from __future__ import annotations

import os
import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmium
import shapely.wkb
from shapely.geometry import box

from osm_cache import OsmExtractCache, snap_bbox_to_grid


class OsmTagHandler(osmium.SimpleHandler):
    """
    Osmium handler that collects features matching specified tags.

    Uses osmium's WKB factory for geometry conversion and supports
    nodes, ways, and areas (multipolygons).
    """

    # Highway types that should be treated as areas when closed
    AREA_HIGHWAY_TYPES = {"pedestrian", "platform", "bus_stop", "elevator"}

    # Tag keys that are inherently area features in OSM
    # When querying for these, closed ways should always be Polygons
    AREA_TAG_KEYS = {
        "natural",      # water, wood, scrub, etc.
        "landuse",      # grass, forest, residential, etc.
        "leisure",      # park, garden, pitch, etc.
        "amenity",      # parking, school, etc.
        "building",     # any building
        "man_made",     # pier, bridge, etc.
        "tourism",      # attraction, camp_site, etc.
        "waterway",     # riverbank (always area when closed)
        "boundary",     # administrative boundaries
        "place",        # islands, squares, etc.
    }

    def __init__(self, tags: dict[str, Any]):
        super().__init__()
        self.tags = tags
        self.wkbfab = osmium.geom.WKBFactory()
        self.features: list[dict[str, Any]] = []
        # Track way IDs processed as areas to avoid duplicates
        self._area_way_ids: set[int] = set()
        # Determine if this query is for area-type features
        self._query_is_area_type = any(k in self.AREA_TAG_KEYS for k in tags.keys())

    def _matches_tags(self, osm_tags: osmium.osm.TagList) -> bool:
        """Check if OSM object tags match any of our filter tags."""
        for key, value in self.tags.items():
            if key not in osm_tags:
                continue
            tag_value = osm_tags[key]
            if value is True:
                return True
            elif isinstance(value, list):
                if tag_value in value:
                    return True
            elif isinstance(value, str):
                if tag_value == value:
                    return True
        return False

    def _extract_tags(self, osm_tags: osmium.osm.TagList) -> dict[str, str]:
        """Extract all tags as a dict."""
        return {tag.k: tag.v for tag in osm_tags}

    def node(self, n: osmium.osm.Node) -> None:
        """Process node features (POIs, etc.)."""
        if not self._matches_tags(n.tags):
            return
        try:
            wkb = self.wkbfab.create_point(n)
            geom = shapely.wkb.loads(wkb, hex=True)
            self.features.append({
                "geometry": geom,
                "osm_type": "node",
                "osm_id": n.id,
                **self._extract_tags(n.tags),
            })
        except Exception:
            pass

    def _should_be_area(self, tags: osmium.osm.TagList) -> bool:
        """
        Check if a closed way should be treated as an area (polygon).

        Uses OSM conventions:
        - Explicit area=yes/no tags override everything
        - Area-type tag keys (natural, leisure, landuse, etc.) are always areas
        - Highway is linear by default, except specific area types
        """
        # Explicit area=yes tag
        if tags.get("area") == "yes":
            return True
        # Explicit area=no means it's a linear feature (e.g., circular road)
        if tags.get("area") == "no":
            return False

        # Check which of our query tags matched this feature
        for query_key, query_value in self.tags.items():
            if query_key not in tags:
                continue

            feature_value = tags[query_key]

            # Check if this query tag matched
            matched = False
            if query_value is True:
                matched = True
            elif isinstance(query_value, list) and feature_value in query_value:
                matched = True
            elif isinstance(query_value, str) and feature_value == query_value:
                matched = True

            if not matched:
                continue

            # This tag matched - determine if it's an area type
            if query_key in self.AREA_TAG_KEYS:
                return True

            # Special handling for highway - only specific types are areas
            if query_key == "highway" and feature_value in self.AREA_HIGHWAY_TYPES:
                return True

        return False

    def way(self, w: osmium.osm.Way) -> None:
        """Process way features as linestrings."""
        if not self._matches_tags(w.tags):
            return
        # Skip closed ways that should be areas - area handler will process them
        if w.is_closed() and len(w.nodes) >= 4 and self._should_be_area(w.tags):
            return
        try:
            wkb = self.wkbfab.create_linestring(w)
            geom = shapely.wkb.loads(wkb, hex=True)

            self.features.append({
                "geometry": geom,
                "osm_type": "way",
                "osm_id": w.id,
                **self._extract_tags(w.tags),
            })
        except Exception:
            pass

    def area(self, a: osmium.osm.Area) -> None:
        """Process area features (closed ways that should be areas, and relations)."""
        if not self._matches_tags(a.tags):
            return

        # For ways: only process if they should be areas
        if a.from_way() and not self._should_be_area(a.tags):
            return

        try:
            wkb = self.wkbfab.create_multipolygon(a)
            geom = shapely.wkb.loads(wkb, hex=True)
            # Simplify to Polygon if only one polygon
            if geom.geom_type == "MultiPolygon" and len(geom.geoms) == 1:
                geom = geom.geoms[0]

            # Determine if from way or relation
            osm_type = "way" if a.from_way() else "relation"
            osm_id = a.orig_id()

            self.features.append({
                "geometry": geom,
                "osm_type": osm_type,
                "osm_id": osm_id,
                **self._extract_tags(a.tags),
            })
        except Exception:
            pass


class OsmReaderError(Exception):
    """Raised when OSM reading operations fail."""

    pass


def tags_to_cache_key(tags: dict[str, Any]) -> str:
    """
    Convert OSM tags dict to a deterministic cache key suffix.

    Args:
        tags: OSM tags dict like {"highway": True} or {"natural": ["water", "bay"]}

    Returns:
        Deterministic string key representing the tags
    """
    # Sort and serialize tags deterministically
    parts = []
    for key in sorted(tags.keys()):
        value = tags[key]
        if value is True:
            parts.append(key)
        elif isinstance(value, list):
            parts.append(f"{key}={','.join(sorted(str(v) for v in value))}")
        else:
            parts.append(f"{key}={value}")

    tag_str = "_".join(parts)
    # Use hash if tag string is too long
    if len(tag_str) > 50:
        return sha256(tag_str.encode()).hexdigest()[:16]
    return tag_str


class OsmOfflineReader:
    """
    Read OSM features from cached PBF extracts.

    Provides a two-level cache:
    - L1: PBF extracts (via OsmExtractCache)
    - L2: Processed GeoDataFrames (pickle files)
    """

    def __init__(
        self,
        extract_cache: OsmExtractCache,
        processed_cache_dir: Path | str | None = None,
    ):
        """
        Initialize the offline reader.

        Args:
            extract_cache: OsmExtractCache instance for PBF extracts
            processed_cache_dir: Directory for L2 GeoDataFrame cache.
                                 Defaults to extract_cache.cache_dir / "processed"
        """
        self.extract_cache = extract_cache

        if processed_cache_dir is None:
            self.processed_cache_dir = extract_cache.cache_dir / "processed"
        else:
            self.processed_cache_dir = Path(processed_cache_dir)

        self.processed_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_processed_cache_path(
        self,
        bbox: tuple[float, float, float, float],
        tags: dict[str, Any],
    ) -> Path:
        """Get the cache path for a processed GeoDataFrame."""
        from osm_cache import bbox_to_cache_key

        snapped_bbox = snap_bbox_to_grid(bbox, self.extract_cache.grid_size)
        bbox_key = bbox_to_cache_key(snapped_bbox)
        tag_key = tags_to_cache_key(tags)
        return self.processed_cache_dir / f"{bbox_key}_{tag_key}.pkl"

    def get_features(
        self,
        bbox: tuple[float, float, float, float],
        tags: dict[str, Any],
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Get OSM features matching tags within bbox.

        Uses two-level caching:
        1. Check L2 cache (processed GeoDataFrame)
        2. If miss, check L1 cache (PBF extract)
        3. Parse PBF, cache result, return

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            tags: OSM tags to filter, e.g. {"highway": True} or {"natural": ["water"]}
            clip_to_bbox: Whether to clip results to exact bbox (default True)

        Returns:
            GeoDataFrame with matching features
        """
        cache_path = self._get_processed_cache_path(bbox, tags)

        # L2 cache hit
        if cache_path.exists():
            try:
                gdf = self._load_from_cache(cache_path)
                if clip_to_bbox:
                    return self._clip_to_bbox(gdf, bbox)
                return gdf
            except Exception:
                # Cache corrupted, regenerate
                cache_path.unlink(missing_ok=True)

        # L1 cache (PBF extract)
        pbf_path = self.extract_cache.get_extract(bbox)

        # Parse PBF
        gdf = self._parse_pbf(pbf_path, tags)

        # Save to L2 cache
        self._save_to_cache(cache_path, gdf)

        if clip_to_bbox:
            return self._clip_to_bbox(gdf, bbox)
        return gdf

    def _parse_pbf(
        self,
        pbf_path: Path,
        tags: dict[str, Any],
    ) -> gpd.GeoDataFrame:
        """
        Parse PBF file and filter by tags using pyosmium.

        Uses pyosmium for full tag access on all geometry types,
        avoiding GDAL/pyogrio schema limitations.

        Args:
            pbf_path: Path to PBF file
            tags: OSM tags to filter

        Returns:
            GeoDataFrame with matching features
        """
        try:
            handler = OsmTagHandler(tags)

            # Apply handler with node location index for way geometry construction
            handler.apply_file(
                str(pbf_path),
                locations=True,  # Enable node location index for ways
                idx="flex_mem",  # Memory-efficient index
            )

            if not handler.features:
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            gdf = gpd.GeoDataFrame(handler.features, crs="EPSG:4326")
            return gdf

        except Exception as e:
            raise OsmReaderError(f"Failed to parse PBF {pbf_path}: {e}") from e

    def _clip_to_bbox(
        self,
        gdf: gpd.GeoDataFrame,
        bbox: tuple[float, float, float, float],
    ) -> gpd.GeoDataFrame:
        """
        Clip GeoDataFrame to exact bbox.

        Args:
            gdf: GeoDataFrame to clip
            bbox: (west, south, east, north) in WGS84 degrees

        Returns:
            Clipped GeoDataFrame
        """
        if gdf.empty:
            return gdf

        west, south, east, north = bbox
        clip_box = box(west, south, east, north)

        return gdf.clip(clip_box)

    def _load_from_cache(self, cache_path: Path) -> gpd.GeoDataFrame:
        """Load GeoDataFrame from pickle cache."""
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def _save_to_cache(self, cache_path: Path, gdf: gpd.GeoDataFrame) -> None:
        """Save GeoDataFrame to pickle cache."""
        # Use temp file for atomic write
        temp_path = cache_path.with_suffix(".tmp.pkl")
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(gdf, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(cache_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def get_roads(
        self,
        bbox: tuple[float, float, float, float],
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Get road network within bbox.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            clip_to_bbox: Whether to clip to exact bbox

        Returns:
            GeoDataFrame with road features
        """
        return self.get_features(
            bbox,
            tags={"highway": True},
            clip_to_bbox=clip_to_bbox,
        )

    def get_water(
        self,
        bbox: tuple[float, float, float, float],
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Get water features within bbox.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            clip_to_bbox: Whether to clip to exact bbox

        Returns:
            GeoDataFrame with water features
        """
        return self.get_features(
            bbox,
            tags={"natural": ["water", "bay"], "waterway": ["riverbank", "river"]},
            clip_to_bbox=clip_to_bbox,
        )

    def get_parks(
        self,
        bbox: tuple[float, float, float, float],
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Get park and green space features within bbox.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            clip_to_bbox: Whether to clip to exact bbox

        Returns:
            GeoDataFrame with park features
        """
        return self.get_features(
            bbox,
            tags={"leisure": "park", "landuse": "grass"},
            clip_to_bbox=clip_to_bbox,
        )

    def get_coastlines(
        self,
        bbox: tuple[float, float, float, float],
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get coastline features within bbox."""
        return self.get_features(
            bbox,
            tags={"natural": "coastline"},
            clip_to_bbox=clip_to_bbox,
        )

    def get_borders(
        self,
        bbox: tuple[float, float, float, float],
        admin_level: int,
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Get administrative border features within bbox.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            admin_level: OSM admin_level (2=country, 4=state, 6=county)
            clip_to_bbox: Whether to clip to exact bbox

        Returns:
            GeoDataFrame with border features
        """
        return self.get_features(
            bbox,
            tags={"boundary": "administrative", "admin_level": str(admin_level)},
            clip_to_bbox=clip_to_bbox,
        )

    def get_glaciers(
        self,
        bbox: tuple[float, float, float, float],
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get glacier features within bbox."""
        return self.get_features(
            bbox,
            tags={"natural": "glacier"},
            clip_to_bbox=clip_to_bbox,
        )

    def get_terrain(
        self,
        bbox: tuple[float, float, float, float],
        clip_to_bbox: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get terrain features (bare rock, scree, cliffs, etc.) within bbox."""
        return self.get_features(
            bbox,
            tags={"natural": ["bare_rock", "scree", "fell", "tundra", "cliff", "rock"]},
            clip_to_bbox=clip_to_bbox,
        )

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about both cache levels."""
        l1_info = self.extract_cache.get_cache_info()
        l2_files = list(self.processed_cache_dir.glob("*.pkl"))

        return {
            "l1_extracts": l1_info,
            "l2_processed": {
                "count": len(l2_files),
                "total_size_mb": sum(f.stat().st_size for f in l2_files)
                // (1024 * 1024),
            },
        }
