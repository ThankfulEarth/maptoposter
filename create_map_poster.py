from matplotlib.figure import Figure
from networkx import MultiDiGraph
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
import sys
from datetime import datetime
import argparse
import pickle
import asyncio
from pathlib import Path
from hashlib import md5
from typing import cast
from geopandas import GeoDataFrame
import geopandas as gpd
import pickle
import math
from shapely.geometry import Point, box

# Offline OSM mode configuration
# NOTE: OSM_PLANET must point to a full planet/regional PBF with ALL features (roads, water, parks, etc.)
# Do NOT use filtered extracts (e.g., boundaries-only) as they will be missing required map data.
USE_OFFLINE_OSM = os.environ.get("OSM_OFFLINE", "false").lower() == "true"
OSM_PLANET_PATH = Path(os.environ.get("OSM_PLANET", "/app/osm/planet-latest.osm.pbf"))
OSM_EXTRACT_CACHE_DIR = Path(os.environ.get("OSM_EXTRACT_CACHE", "/app/osm/extracts"))

# Land polygons for land/water distinction
# Full resolution > simplified > Natural Earth 10m > none
_LAND_FULL = Path(__file__).parent / "data" / "osm_land" / "land-polygons-complete-3857" / "land_polygons.shp"
_LAND_SIMPLIFIED = Path(__file__).parent / "data" / "osm_land" / "simplified-land-polygons-complete-3857" / "simplified_land_polygons.shp"
_LAND_NE = Path(__file__).parent / "data" / "natural_earth" / "ne_10m_land.shp"
_land_gdf = None
_land_source = None
_land_loaded = False


def _get_land_polygons(clip_bbox=None):
    """Load land polygons with bbox-based reading for large datasets.

    For the full-resolution dataset (~800MB), uses gpd.read_file(bbox=...)
    to only load polygons intersecting the clip area. Result is NOT cached
    since each poster has a different bbox.

    For smaller datasets (simplified, Natural Earth), loads everything once
    and caches globally.
    """
    global _land_gdf, _land_source, _land_loaded

    # Full-resolution: always read with bbox filter (too large to load entirely)
    # The shapefile is EPSG:3857 so we must reproject the WGS84 bbox
    if _LAND_FULL.exists():
        if clip_bbox is not None:
            from pyproj import Transformer
            west, south, east, north = clip_bbox
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(west, south)
            x_max, y_max = transformer.transform(east, north)
            gdf = gpd.read_file(_LAND_FULL, bbox=(x_min, y_min, x_max, y_max))
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
        # No clip_bbox — fall through to simplified/NE

    # Simplified or Natural Earth: load once, cache globally
    if _land_loaded:
        return _land_gdf
    _land_loaded = True
    for path in (_LAND_SIMPLIFIED, _LAND_NE):
        if path.exists():
            gdf = gpd.read_file(path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            _land_gdf = gdf
            _land_source = str(path)
            return _land_gdf
    return None


# Lazy-loaded offline reader (only initialized if USE_OFFLINE_OSM is True)
_offline_reader = None


def _get_offline_reader():
    """Get or create the offline OSM reader singleton."""
    global _offline_reader
    if _offline_reader is None:
        from osm_cache import OsmExtractCache
        from osm_reader import OsmOfflineReader

        cache = OsmExtractCache(OSM_PLANET_PATH, OSM_EXTRACT_CACHE_DIR)
        _offline_reader = OsmOfflineReader(cache)
    return _offline_reader


def point_dist_to_bbox(
    point: tuple[float, float], dist: float
) -> tuple[float, float, float, float]:
    """
    Convert a center point and distance to a bounding box.

    Uses approximate conversion: 1 degree latitude ~ 111km.
    Longitude varies with latitude.

    Args:
        point: (latitude, longitude) tuple
        dist: Distance in meters from center to edge

    Returns:
        (west, south, east, north) bbox in degrees
    """
    lat, lon = point
    # Approximate degrees per meter
    lat_deg_per_m = 1 / 111000
    lon_deg_per_m = 1 / (111000 * abs(np.cos(np.radians(lat))))

    lat_delta = dist * lat_deg_per_m
    lon_delta = dist * lon_deg_per_m

    return (
        lon - lon_delta,  # west
        lat - lat_delta,  # south
        lon + lon_delta,  # east
        lat + lat_delta,  # north
    )

class CacheError(Exception):
    """Raised when a cache operation fails."""
    pass

CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)


THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

CACHE_DIR = ".cache"

class CacheError(Exception):
    pass


def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}")


def cache_set(key: str, value):
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}")


def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    
    # Verify fonts exist
    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None
    
    return fonts

FONTS = load_fonts()


def resolve_title_fonts(title_font_id, fonts_root: str = FONTS_DIR) -> dict:
    """Return {bold,regular,light} ttf paths for the title font, or bundled Roboto."""
    roboto = {
        "bold": os.path.join(FONTS_DIR, "Roboto-Bold.ttf"),
        "regular": os.path.join(FONTS_DIR, "Roboto-Regular.ttf"),
        "light": os.path.join(FONTS_DIR, "Roboto-Light.ttf"),
    }
    if not title_font_id:
        return roboto
    gen = os.path.join(fonts_root, "generated", title_font_id)
    resolved = {role: os.path.join(gen, f"{role}.ttf") for role in ("bold", "regular", "light")}
    if all(os.path.exists(p) for p in resolved.values()):
        return resolved
    return roboto


def generate_output_filename(city, theme_name, output_format):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_')
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes

def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        # Fallback to embedded default theme
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }
    
    with open(theme_file, 'r') as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if 'description' in theme:
            print(f"  {theme['description']}")
        return theme

# Load theme (can be changed via command line or input)
THEME = dict[str, str]()  # Will be loaded later

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def get_edge_colors_by_type(G):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []
    
    for u, v, data in G.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')
        
        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign color based on road type
        if highway in ['motorway', 'motorway_link']:
            color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']:
            color = THEME['road_residential']
        else:
            color = THEME['road_default']
        
        edge_colors.append(color)
    
    return edge_colors

def get_edge_widths_by_type(G):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign width based on road importance
        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4
        
        edge_widths.append(width)
    
    return edge_widths

def classify_highway(highway):
    """Normalize highway value (can be a list or string) to a single string."""
    if isinstance(highway, list):
        highway = highway[0] if highway else 'unclassified'
    if not isinstance(highway, str):
        return 'unclassified'
    return highway


def get_road_color(highway):
    """Get theme color for a highway type."""
    highway = classify_highway(highway)
    if highway in ('motorway', 'motorway_link'):
        return THEME['road_motorway']
    if highway in ('trunk', 'trunk_link', 'primary', 'primary_link'):
        return THEME['road_primary']
    if highway in ('secondary', 'secondary_link'):
        return THEME['road_secondary']
    if highway in ('tertiary', 'tertiary_link'):
        return THEME['road_tertiary']
    if highway in ('residential', 'living_street', 'unclassified'):
        return THEME['road_residential']
    return THEME['road_default']


def get_road_width(highway):
    """Get line width for a highway type."""
    highway = classify_highway(highway)
    if highway in ('motorway', 'motorway_link'):
        return 1.2
    if highway in ('trunk', 'trunk_link', 'primary', 'primary_link'):
        return 1.0
    if highway in ('secondary', 'secondary_link'):
        return 0.8
    if highway in ('tertiary', 'tertiary_link'):
        return 0.6
    return 0.4


def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}")

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError:
            # If an event loop is already running, try using it to complete the coroutine.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running event loop in the same thread; raise a clear error.
                raise RuntimeError("Geocoder returned a coroutine while an event loop is already running. Run this script in a synchronous environment.")
            location = loop.run_until_complete(location)
    
    if location:
        # Use getattr to safely access address (helps static analyzers)
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        else:
            print("✓ Found location (address not available)")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")
    
def get_crop_limits(crs, center_lat_lon, fig, dist, bbox=None):
    """
    Crop inward to preserve aspect ratio while guaranteeing
    full coverage of the requested area.

    Args:
        crs: The target CRS (from graph or UTM zone)
        center_lat_lon: (lat, lon) tuple
        fig: matplotlib Figure
        dist: distance in meters (used when bbox is None)
        bbox: optional (west, south, east, north) in WGS84
    """
    lat, lon = center_lat_lon

    # Project center point into target CRS
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=crs
        )[0]
    )
    center_x, center_y = center.x, center.y

    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height

    if bbox is not None:
        west, south, east, north = bbox
        # Project bbox corners into target CRS to get metric extents
        sw = ox.projection.project_geometry(
            Point(west, south), crs="EPSG:4326", to_crs=crs
        )[0]
        ne = ox.projection.project_geometry(
            Point(east, north), crs="EPSG:4326", to_crs=crs
        )[0]
        half_x = max(abs(ne.x - center_x), abs(center_x - sw.x))
        half_y = max(abs(ne.y - center_y), abs(center_y - sw.y))

        # Expand the smaller axis so the full bbox fits within the poster
        if half_x / half_y > aspect:
            half_y = half_x / aspect
        else:
            half_x = half_y * aspect
    else:
        # Start from the *requested* radius
        half_x = dist
        half_y = dist

        # Cut inward to match aspect
        if aspect > 1:  # landscape → reduce height
            half_y = half_x / aspect
        else:           # portrait → reduce width
            half_x = half_y * aspect

    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


ROAD_DETAIL_LEVELS = {
    "low": [
        'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
    ],
    "medium": [
        'motorway', 'motorway_link', 'trunk', 'trunk_link',
        'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link',
    ],
    "high": [
        'motorway', 'motorway_link', 'trunk', 'trunk_link',
        'primary', 'primary_link', 'secondary', 'secondary_link',
        'tertiary', 'tertiary_link', 'residential', 'living_street', 'unclassified', 'service',
    ],
}


def resolve_road_detail(road_detail: str, bbox=None) -> str:
    """Auto-detect road detail level from bbox area when set to 'auto'."""
    if road_detail != "auto":
        return road_detail
    if bbox is None:
        return "high"
    west, south, east, north = bbox
    # Approximate area in km² using lat/lon deltas
    lat_mid = (south + north) / 2
    width_km = (east - west) * 111 * math.cos(math.radians(lat_mid))
    height_km = (north - south) * 111
    area_km2 = width_km * height_km
    if area_km2 > 500_000:
        level = "low"
    elif area_km2 > 500:
        level = "medium"
    else:
        level = "high"
    print(f"  Road detail: {level} (bbox area ≈ {area_km2:,.0f} km²)")
    return level


def utm_crs_from_point(lat: float, lon: float) -> str:
    """Derive the UTM EPSG code for a center point, independent of any road data."""
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    return f"EPSG:{epsg}"


def fetch_roads_offline(bbox, road_detail: str = "high") -> GeoDataFrame | None:
    """
    Fetch road geometries from offline PBF extracts.

    Returns a GeoDataFrame with highway LineStrings instead of an OSMnx graph.
    Uses road_detail level to filter highway types (low/medium/high).
    """
    west, south, east, north = bbox
    cache_key = f"roads_offline_{road_detail}_bbox_{north}_{south}_{east}_{west}"

    cached = cache_get(cache_key)
    if cached is not None:
        print("✓ Using cached offline roads")
        return cast(GeoDataFrame, cached)

    highway_types = ROAD_DETAIL_LEVELS.get(road_detail, ROAD_DETAIL_LEVELS["high"])

    try:
        reader = _get_offline_reader()
        roads = reader.get_features(bbox, tags={"highway": highway_types}, clip_to_bbox=True)
        if roads is None or roads.empty:
            print("⚠ No roads found in offline PBF for this bbox")
            return None

        # Filter to LineString/MultiLineString geometries only
        roads = roads[roads.geometry.type.isin(['LineString', 'MultiLineString'])]

        print(f"✓ Loaded {len(roads)} roads from offline OSM ({road_detail} detail)")
        try:
            cache_set(cache_key, roads)
        except CacheError as e:
            print(e)
        return roads
    except Exception as e:
        print(f"Offline OSM error while fetching roads: {e}")
        return None


def fetch_graph(point, dist, bbox=None) -> MultiDiGraph | None:
    """
    Fetch street network graph for a point+distance or bbox.

    In online mode, uses OSMnx/Overpass API.
    In offline mode with bbox, use fetch_roads_offline() instead (called from create_poster).
    """
    if bbox is not None:
        west, south, east, north = bbox
        graph = f"graph_bbox_{north}_{south}_{east}_{west}"
    else:
        lat, lon = point
        graph = f"graph_{lat}_{lon}_{dist}"

    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        if bbox is not None:
            west, south, east, north = bbox
            G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='all', truncate_by_edge=True)
        else:
            G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all', truncate_by_edge=True)
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, G)
        except CacheError as e:
            print(e)
        return G
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None

def fetch_features(point, dist, tags, name, bbox=None) -> GeoDataFrame | None:
    """
    Fetch OSM features for a point+distance or bbox.

    In offline mode, uses local PBF extracts instead of Overpass API.
    """
    if bbox is not None:
        west, south, east, north = bbox
        tag_str = "_".join(tags.keys())
        features = f"{name}_bbox_{north}_{south}_{east}_{west}_{tag_str}"
    else:
        lat, lon = point
        tag_str = "_".join(tags.keys())
        features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"

    # Check local pickle cache first (works for both modes)
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        if bbox is not None:
            if USE_OFFLINE_OSM:
                reader = _get_offline_reader()
                data = reader.get_features(bbox, tags, clip_to_bbox=True)
                print(f"✓ Loaded {name} from offline OSM (bbox extract)")
            else:
                west, south, east, north = bbox
                data = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
                time.sleep(0.3)
        elif USE_OFFLINE_OSM:
            # Offline mode: use local PBF extracts
            reader = _get_offline_reader()
            point_bbox = point_dist_to_bbox(point, dist)
            data = reader.get_features(point_bbox, tags, clip_to_bbox=True)
            print(f"✓ Loaded {name} from offline OSM (bbox extract)")
        else:
            # Online mode: use OSMnx/Overpass API
            data = ox.features_from_point(point, tags=tags, dist=dist)
            # Rate limit between requests
            time.sleep(0.3)

        # Cache the result (pickle cache for fast subsequent loads)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        mode = "offline" if USE_OFFLINE_OSM else "OSMnx"
        print(f"{mode} error while fetching features: {e}")
        return None



def resolve_layers(no_water: bool, no_parks: bool, no_roads: bool) -> dict:
    """Map --no-* flags to a draw/skip decision per layer (True = draw)."""
    return {"water": not no_water, "parks": not no_parks, "roads": not no_roads}


def buildings_allowed(area_km2: float, cap_km2: float = 200.0) -> bool:
    """Buildings are fetched only for bboxes at or under the area cap."""
    return area_km2 <= cap_km2


def create_poster(city, country, point, dist, output_file, output_format, width=12, height=16, country_label=None, name_label=None, dpi=300, brand=None, coastline=False, borders_level=None, glaciers=False, terrain=False, bbox=None, road_detail="auto", draw_water=True, draw_parks=True, draw_roads=True, draw_buildings=False, title_font=None, progress_callback=None):
    print(f"\nGenerating map for {city}, {country}...")

    # When bbox is provided, expand to cover the aspect-ratio-adjusted crop area.
    # For city-level bboxes, expand fully so roads fill the poster frame.
    # For large bboxes (countries), use modest padding — empty areas are ocean/bg.
    fetch_bbox = None
    if bbox is not None:
        west, south, east, north = bbox
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2
        half_lon = (east - west) / 2
        half_lat = (north - south) / 2
        # Estimate bbox area in km²
        width_km = (east - west) * 111 * abs(math.cos(math.radians(center_lat)))
        height_km = (north - south) * 111
        area_km2 = width_km * height_km

        # Expand only the dimension that's too short for the poster aspect ratio
        poster_aspect = width / height  # e.g. 0.75 for 6x8
        cos_lat = abs(math.cos(math.radians(center_lat)))
        bbox_width_km = half_lon * 2 * 111 * cos_lat
        bbox_height_km = half_lat * 2 * 111
        bbox_aspect = bbox_width_km / bbox_height_km

        pad = 0.1  # 10% buffer on untouched dimension
        if bbox_aspect > poster_aspect:
            # bbox wider than needed → expand height only
            target_half_lat = (half_lon * cos_lat) / poster_aspect / 111 / 2 * 2
            # Simpler: target_height_km = bbox_width_km / poster_aspect
            target_half_lat = bbox_width_km / poster_aspect / 2 / 111
            fetch_bbox = (
                west - half_lon * pad,
                center_lat - target_half_lat * (1 + pad),
                east + half_lon * pad,
                center_lat + target_half_lat * (1 + pad),
            )
        else:
            # bbox taller than needed → expand width only
            target_half_lon = bbox_height_km * poster_aspect / 2 / 111 / cos_lat
            fetch_bbox = (
                center_lon - target_half_lon * (1 + pad),
                south - half_lat * pad,
                center_lon + target_half_lon * (1 + pad),
                north + half_lat * pad,
            )

    # Calculate total steps for progress bar
    total_steps = 4  # street network, water, parks, buildings
    if coastline:
        total_steps += 1
    if borders_level is not None:
        total_steps += 1
    if glaciers:
        total_steps += 1
    if terrain:
        total_steps += 1
    total_steps += 3  # renderingMap, applyingStyles, savingPoster

    current_step = 0

    def report_progress(step_id, output_path=None):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(step_id, current_step, total_steps, output_path=output_path)

    # Decide whether to use offline roads (GeoDataFrame) or online graph
    use_offline_roads = USE_OFFLINE_OSM and fetch_bbox is not None
    resolved_road_detail = resolve_road_detail(road_detail, bbox=bbox)

    # Compensated distance is needed for feature fetch radius and crop limits
    # regardless of whether roads are drawn, so it's computed unconditionally.
    compensated_dist = dist * (max(height, width) / min(height, width)) / 4  # To compensate for viewport crop

    # Progress bar for data fetching
    with tqdm(total=total_steps, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        G = None
        roads_gdf = None
        if draw_roads:
            if use_offline_roads:
                pbar.set_description("Loading roads from offline OSM")
                roads_gdf = fetch_roads_offline(fetch_bbox, road_detail=resolved_road_detail)
                if roads_gdf is None or roads_gdf.empty:
                    raise RuntimeError("Failed to retrieve road data from offline PBF.")
            else:
                pbar.set_description("Downloading street network")
                if fetch_bbox is not None:
                    G = fetch_graph(point, dist, bbox=fetch_bbox)
                else:
                    G = fetch_graph(point, compensated_dist)
                if G is None:
                    raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)
        report_progress("fetchingRoads")

        # For point+dist mode, use compensated_dist for features; for bbox mode, use fetch_bbox
        feat_dist = None if fetch_bbox else compensated_dist

        # 2. Fetch Water Features (inland only — bays removed, they create dark ocean blobs)
        water = None
        if draw_water:
            pbar.set_description("Downloading water features")
            water = fetch_features(point, feat_dist, tags={'natural': 'water', 'waterway': 'riverbank'}, name='water', bbox=fetch_bbox)
        pbar.update(1)
        report_progress("fetchingWater")

        # 3. Fetch Parks
        parks = None
        if draw_parks:
            pbar.set_description("Downloading parks/green spaces")
            parks = fetch_features(point, feat_dist, tags={'leisure': 'park', 'landuse': 'grass'}, name='parks', bbox=fetch_bbox)
        pbar.update(1)
        report_progress("fetchingParks")

        # 3b. Fetch Buildings (optional, capped by bbox area)
        buildings = None
        if draw_buildings:
            if bbox is not None and not buildings_allowed(area_km2):
                print(f"⚠ Skipping buildings: bbox area {area_km2:.0f} km² exceeds 200 km² cap")
            else:
                pbar.set_description("Downloading buildings")
                buildings = fetch_features(point, feat_dist, tags={'building': True}, name='buildings', bbox=fetch_bbox)
        report_progress("fetchingBuildings")

        # 4. Fetch Coastlines (optional)
        coastlines_data = None
        if coastline:
            pbar.set_description("Downloading coastlines")
            coastlines_data = fetch_features(point, feat_dist, tags={'natural': 'coastline'}, name='coastlines', bbox=fetch_bbox)
            pbar.update(1)
            report_progress("fetchingCoastlines")

        # 5. Fetch Borders (optional)
        borders_data = None
        if borders_level is not None:
            pbar.set_description(f"Downloading admin borders (level {borders_level})")
            borders_data = fetch_features(point, feat_dist, tags={'boundary': 'administrative', 'admin_level': str(borders_level)}, name=f'borders_L{borders_level}', bbox=fetch_bbox)
            pbar.update(1)
            report_progress("fetchingBorders")

        # 6. Fetch Glaciers (optional)
        glaciers_data = None
        if glaciers:
            pbar.set_description("Downloading glaciers")
            glaciers_data = fetch_features(point, feat_dist, tags={'natural': 'glacier'}, name='glaciers', bbox=fetch_bbox)
            pbar.update(1)
            report_progress("fetchingGlaciers")

        # 7. Fetch Terrain (optional)
        terrain_data = None
        if terrain:
            pbar.set_description("Downloading terrain features")
            terrain_data = fetch_features(point, feat_dist, tags={'natural': ['bare_rock', 'scree', 'fell', 'tundra', 'cliff', 'rock']}, name='terrain', bbox=fetch_bbox)
            pbar.update(1)
            report_progress("fetchingTerrain")

    print("✓ All data retrieved successfully!")
    
    # 2. Setup Plot — background is water color, land polygons fill landmass
    print("Rendering map...")
    report_progress("renderingMap")
    water_bg = THEME.get('water', THEME['bg'])
    fig, ax = plt.subplots(figsize=(width, height), facecolor=water_bg)
    ax.set_facecolor(water_bg)
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    # Project to a metric CRS so distances and aspect are linear (meters)
    G_proj = None
    roads_proj = None
    lat, lon = point
    if not draw_roads:
        # No road data was fetched — derive the CRS from the center point's UTM
        # zone directly, independent of any road graph/GeoDataFrame.
        target_crs = utm_crs_from_point(lat, lon)
    elif use_offline_roads:
        # Derive UTM CRS from center point (no graph available)
        target_crs = utm_crs_from_point(lat, lon)
        roads_proj = roads_gdf.to_crs(target_crs)
    else:
        G_proj = ox.project_graph(G)
        target_crs = G_proj.graph["crs"]

    def project_features(gdf):
        """Project a GeoDataFrame to the target CRS."""
        try:
            return ox.projection.project_gdf(gdf)
        except Exception:
            return gdf.to_crs(target_crs)

    # 3. Plot Layers
    # Layer 0: Land polygons — fills landmass over water background
    land_clip_bbox = fetch_bbox if fetch_bbox else (lon - 1, lat - 1, lon + 1, lat + 1)
    land_gdf = _get_land_polygons(clip_bbox=land_clip_bbox)
    if land_gdf is not None:
        clip_box = box(land_clip_bbox[0], land_clip_bbox[1], land_clip_bbox[2], land_clip_bbox[3])
        land_clipped = land_gdf.clip(clip_box)
        if not land_clipped.empty:
            land_proj = project_features(land_clipped)
            land_proj.plot(ax=ax, facecolor=THEME['bg'], edgecolor='none', zorder=0)

    if water is not None and not water.empty:
        water_polys = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polys.empty:
            water_polys = project_features(water_polys)
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)

    if parks is not None and not parks.empty:
        parks_polys = parks[parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not parks_polys.empty:
            parks_polys = project_features(parks_polys)
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)

    if buildings is not None and not buildings.empty:
        building_polys = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not building_polys.empty:
            building_polys = project_features(building_polys)
            building_color = THEME.get('building', THEME.get('road_residential', THEME['text']))
            building_polys.plot(ax=ax, facecolor=building_color, edgecolor='none', zorder=2.5)

    # Layer 1a: Glaciers (optional)
    if glaciers_data is not None and not glaciers_data.empty:
        glaciers_polys = glaciers_data[glaciers_data.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not glaciers_polys.empty:
            glaciers_polys = project_features(glaciers_polys)
            glacier_color = THEME.get('glaciers', '#CEEAEE')
            glaciers_polys.plot(ax=ax, facecolor=glacier_color, edgecolor='none', zorder=0)

    # Layer 1a2: Terrain (optional)
    if terrain_data is not None and not terrain_data.empty:
        terrain_polys = terrain_data[terrain_data.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not terrain_polys.empty:
            terrain_polys = project_features(terrain_polys)
            terrain_color = THEME.get('terrain', '#D4C4B0')
            terrain_polys.plot(ax=ax, facecolor=terrain_color, edgecolor='none', zorder=1)

    # Layer 1b: Coastlines (optional)
    if coastlines_data is not None and not coastlines_data.empty:
        coastline_lines = coastlines_data[coastlines_data.geometry.type.isin(['LineString', 'MultiLineString'])]
        if not coastline_lines.empty:
            coastline_lines = project_features(coastline_lines)
            coastline_color = THEME.get('coastline', THEME['text'])
            coastline_width = 0.5 * {"low": 0.3, "medium": 0.4, "high": 1.0}.get(resolved_road_detail, 1.0)
            coastline_lines.plot(ax=ax, edgecolor=coastline_color, linewidth=coastline_width, zorder=3)

    # Layer 1c: Administrative borders (optional)
    if borders_data is not None and not borders_data.empty:
        border_lines = borders_data[borders_data.geometry.type.isin(['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon'])]
        if not border_lines.empty:
            border_lines = project_features(border_lines)
            border_color = THEME.get('borders', THEME['text'])
            border_lines.plot(ax=ax, facecolor='none', edgecolor=border_color, linewidth=0.6, linestyle='--', zorder=3)

    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    report_progress("applyingStyles")
    crop_xlim, crop_ylim = get_crop_limits(target_crs, point, fig, feat_dist if feat_dist else dist, bbox=bbox)

    if draw_roads:
        if use_offline_roads:
            # Scale road widths by detail level (thinner at country zoom)
            width_scale = {"low": 0.3, "medium": 0.4, "high": 1.0}.get(resolved_road_detail, 1.0)
            colors = roads_proj['highway'].apply(get_road_color)
            widths = roads_proj['highway'].apply(lambda h: get_road_width(h) * width_scale)
            for (color, width_val), group in roads_proj.groupby([colors, widths]):
                group.plot(ax=ax, edgecolor=color, linewidth=width_val, zorder=4)
            # Hide axis decorations (ox.plot_graph does this automatically)
            ax.set_axis_off()
        else:
            # Plot roads from OSMnx graph (online mode)
            edge_colors = get_edge_colors_by_type(G_proj)
            edge_widths = get_edge_widths_by_type(G_proj)
            ox.plot_graph(
                G_proj, ax=ax, bgcolor=THEME['bg'],
                node_size=0,
                edge_color=edge_colors,
                edge_linewidth=edge_widths,
                show=False, close=False
            )
    else:
        # No road graph/GeoDataFrame to plot — still hide axis decorations
        # (ox.plot_graph would otherwise handle this automatically).
        ax.set_axis_off()

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)
    
    # Layer 3: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)
    
    # Calculate scale factor based on poster width (reference width 12 inches)
    scale_factor = width / 12.0
    
    # Base font sizes (at 12 inches width)
    BASE_MAIN = 60
    BASE_TOP = 40
    BASE_SUB = 22
    BASE_COORDS = 14

    # 4. Typography using the resolved title font (falls back to bundled Roboto)
    title_fonts = resolve_title_fonts(title_font)
    font_main = FontProperties(fname=title_fonts['bold'], size=BASE_MAIN * scale_factor)
    font_top = FontProperties(fname=title_fonts['bold'], size=BASE_TOP * scale_factor)
    font_sub = FontProperties(fname=title_fonts['light'], size=BASE_SUB * scale_factor)
    font_coords = FontProperties(fname=title_fonts['regular'], size=BASE_COORDS * scale_factor)

    spaced_city = "  ".join(list(city.upper()))
    
    # Dynamically adjust font size based on city name length to prevent truncation
    # We use the already scaled "main" font size as the starting point.
    base_adjusted_main = BASE_MAIN * scale_factor
    city_char_count = len(city)
    
    # Heuristic: If length is > 10, start reducing.
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor) 
    else:
        adjusted_font_size = base_adjusted_main
    
    font_main_adjusted = FontProperties(fname=title_fonts['bold'], size=adjusted_font_size)

    # --- BOTTOM TEXT ---
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_main_adjusted, zorder=11)
    
    country_text = country_label if country_label is not None else country
    ax.text(0.5, 0.10, country_text.upper(), transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)
    
    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")
    
    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)
    
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
            color=THEME['text'], linewidth=1 * scale_factor, zorder=11)

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_attr = FontProperties(family='monospace', size=8)
    
    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=THEME['text'], alpha=0.5, ha='right', va='bottom',
            fontproperties=font_attr, zorder=11)

    # --- BRAND (bottom left) ---
    if brand:
        ax.text(0.02, 0.02, brand, transform=ax.transAxes,
                color=THEME['text'], alpha=0.5, ha='left', va='bottom',
                fontproperties=font_attr, zorder=11)

    # 5. Save
    print(f"Saving to {output_file}...")
    report_progress("savingPoster")

    fmt = output_format.lower()
    save_kwargs = dict(facecolor=water_bg, pad_inches=0,)

    # DPI matters mainly for raster formats
    if fmt == "png":
        save_kwargs["dpi"] = dpi

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")

    if progress_callback:
        progress_callback("complete", total_steps, total_steps, output_path=output_file)


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid
  
  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline
  
  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads
  
  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout
  
  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  
  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split
  
  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --country-label   Override country text displayed on poster
  --theme, -t       Theme name (default: feature_based)
  --all-themes      Generate posters for all themes
  --distance, -d    Map radius in meters (default: 29000)
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """
    )
    
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--country-label', dest='country_label', type=str, help='Override country text displayed on poster')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--all-themes', '--All-themes', dest='all_themes', action='store_true', help='Generate posters for all themes')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--bbox', type=str, help='Bounding box as west,south,east,north (decimal degrees). Overrides --distance')
    parser.add_argument('--width', '-W', type=float, default=12, help='Image width in inches (default: 12)')
    parser.add_argument('--height', '-H', type=float, default=16, help='Image height in inches (default: 16)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'svg', 'pdf'],help='Output format for the poster (default: png)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PNG output (default: 300)')
    parser.add_argument('--lat', type=float, help='Latitude (use with --lon to skip geocoding)')
    parser.add_argument('--lon', type=float, help='Longitude (use with --lat to skip geocoding)')
    parser.add_argument('--brand', type=str, help='Brand text to display in bottom left corner')
    parser.add_argument('--coastline', action='store_true', help='Show coastlines')
    parser.add_argument('--borders', type=int, metavar='LEVEL', help='Show administrative borders (2=country, 4=state/region, 6=county)')
    parser.add_argument('--glaciers', action='store_true', help='Show glaciers/ice sheets')
    parser.add_argument('--experimental-terrain', dest='terrain', action='store_true', help='Show terrain features (bare rock, scree, cliffs, tundra)')
    parser.add_argument('--road-detail', default='auto', choices=['auto', 'low', 'medium', 'high'], help='Road detail level (default: auto, based on bbox area)')
    parser.add_argument('--progress', action='store_true', help='Emit machine-readable PROGRESS: lines to stdout')
    parser.add_argument('--colors-json', dest='colors_json', type=str, help='JSON string with full color overrides (snake_case keys). Skips theme file loading.')
    parser.add_argument('--no-water', dest='no_water', action='store_true', help='Disable water features')
    parser.add_argument('--no-parks', dest='no_parks', action='store_true', help='Disable park features')
    parser.add_argument('--no-roads', dest='no_roads', action='store_true', help='Disable road features')
    parser.add_argument('--buildings', dest='buildings', action='store_true', help='Enable building footprints (capped at 200km² bbox area)')
    parser.add_argument('--title-font', dest='title_font', type=str, help='Font ID for poster title typography')

    args = parser.parse_args()
    
    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)
    
    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)

    # Parse bbox if provided
    parsed_bbox = None
    if args.bbox:
        try:
            parts = [float(x.strip()) for x in args.bbox.split(',')]
            if len(parts) != 4:
                raise ValueError("Expected 4 values")
            parsed_bbox = tuple(parts)  # (west, south, east, north)
        except ValueError:
            print("Error: --bbox must be 4 comma-separated numbers: west,south,east,north\n")
            sys.exit(1)

    # Validate lat/lon: if one is provided, both must be provided
    if (args.lat is None) != (args.lon is None):
        print("Error: --lat and --lon must be used together.\n")
        sys.exit(1)

    # When bbox is provided, derive center coords from it
    has_coordinates = args.lat is not None and args.lon is not None
    if parsed_bbox and not has_coordinates:
        west, south, east, north = parsed_bbox
        args.lat = (south + north) / 2
        args.lon = (west + east) / 2
        has_coordinates = True

    # Validate required arguments
    if not args.city:
        print("Error: --city is required.\n")
        print_examples()
        sys.exit(1)
    if not has_coordinates and not args.country:
        print("Error: --country is required (unless using --lat/--lon or --bbox).\n")
        print_examples()
        sys.exit(1)
    
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        os.sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    elif args.colors_json:
        themes_to_generate = [args.theme]
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            os.sys.exit(1)
        themes_to_generate = [args.theme]
    
    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    def _progress_callback(step_id, step, total_steps, output_path=None):
        obj = {"stepId": step_id, "step": step, "totalSteps": total_steps}
        if output_path is not None:
            obj["outputPath"] = str(output_path)
        print(f"PROGRESS:{json.dumps(obj)}", flush=True)

    progress_cb = _progress_callback if args.progress else None

    # Get coordinates and generate poster
    try:
        if args.lat is not None and args.lon is not None:
            coords = (args.lat, args.lon)
            print(f"Using provided coordinates: {args.lat}, {args.lon}")
        else:
            coords = get_coordinates(args.city, args.country)

        country_for_poster = args.country if args.country else ""
        layers = resolve_layers(args.no_water, args.no_parks, args.no_roads)
        for theme_name in themes_to_generate:
            if args.colors_json:
                THEME = json.loads(args.colors_json)
                THEME.setdefault("name", theme_name)
                print(f"✓ Using colors from --colors-json for theme: {theme_name}")
            else:
                THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            create_poster(args.city, country_for_poster, coords, args.distance, output_file, args.format, args.width, args.height, country_label=args.country_label, dpi=args.dpi, brand=args.brand, coastline=args.coastline, borders_level=args.borders, glaciers=args.glaciers, terrain=args.terrain, bbox=parsed_bbox, road_detail=args.road_detail, draw_water=layers["water"], draw_parks=layers["parks"], draw_roads=layers["roads"], draw_buildings=args.buildings, title_font=args.title_font, progress_callback=progress_cb)
        
        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
