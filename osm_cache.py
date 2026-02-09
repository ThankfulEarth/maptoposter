"""
OSM Extract Cache - Two-level caching for offline OSM data.

Level 1: PBF extracts (slow to create ~10 min, reusable across different tag queries)
Level 2: Processed GeoDataFrames (fast to create from L1)

Features:
- Grid-snaps bboxes to 0.05 degrees for better cache reuse
- Automatic source selection: picks smallest available extract covering requested bbox
- Falls back to planet file when no regional extract matches

Regional extracts use bbox-based filenames:
  bbox_{west}_{south}_{east}_{north}[_{hierarchy}].osm.pbf
  e.g., bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf
"""

from __future__ import annotations

import fcntl
import logging
import re
import subprocess
from math import ceil, floor
from pathlib import Path

logger = logging.getLogger(__name__)

GRID_SIZE = 0.05  # ~5.5km at equator, good balance between cache reuse and data size


class OsmCacheError(Exception):
    """Raised when an OSM cache operation fails."""

    pass


def snap_bbox_to_grid(
    bbox: tuple[float, float, float, float], grid: float = GRID_SIZE
) -> tuple[float, float, float, float]:
    """
    Expand bbox to grid boundaries for caching.

    Always expands outward (floor for west/south, ceil for east/north)
    to ensure the snapped bbox fully contains the original.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees
        grid: Grid size in degrees (default 0.05)

    Returns:
        Snapped bbox that fully contains the original
    """
    west, south, east, north = bbox
    return (
        floor(west / grid) * grid,
        floor(south / grid) * grid,
        ceil(east / grid) * grid,
        ceil(north / grid) * grid,
    )


def bbox_to_cache_key(bbox: tuple[float, float, float, float]) -> str:
    """
    Convert bbox to a filesystem-safe cache key.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees

    Returns:
        Cache key string like "bbox_12.4500_55.6000_12.7000_55.7500"
    """
    west, south, east, north = bbox
    return f"bbox_{west:.4f}_{south:.4f}_{east:.4f}_{north:.4f}"


def parse_bbox_from_filename(
    filename: str,
) -> tuple[float, float, float, float] | None:
    """
    Extract bbox from filename.

    Parses filenames like:
    - bbox_7.5000_54.5000_15.5000_58.0000.osm.pbf
    - bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf
    - bbox_-74.0500_40.7000_-73.8500_40.8500_Americas_US_US-NY.osm.pbf

    Args:
        filename: PBF filename (not full path)

    Returns:
        (west, south, east, north) tuple or None if invalid format
    """
    # Skip temp files
    if ".tmp." in filename:
        return None

    match = re.match(r"bbox_(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)", filename)
    if match:
        return (
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )
    return None


def bbox_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    """
    Check if outer bbox fully contains inner bbox.

    Args:
        outer: (west, south, east, north) containing bbox
        inner: (west, south, east, north) contained bbox

    Returns:
        True if outer fully contains inner
    """
    return (
        outer[0] <= inner[0]  # west
        and outer[1] <= inner[1]  # south
        and outer[2] >= inner[2]  # east
        and outer[3] >= inner[3]  # north
    )


class OsmExtractCache:
    """
    Cache manager for OSM PBF extracts.

    Extracts bboxes from a planet file and caches them for reuse.
    Uses file-based locking for concurrent worker safety.
    """

    def __init__(
        self,
        planet_path: Path | str,
        cache_dir: Path | str,
        grid_size: float = GRID_SIZE,
        complete_partial_relations: int = 1,
    ):
        """
        Initialize the extract cache.

        Args:
            planet_path: Path to the source OSM PBF file (planet or regional extract)
            cache_dir: Directory for storing cached extracts
            grid_size: Grid size for bbox snapping (default 0.05 degrees)
            complete_partial_relations: Percentage threshold for completing partial relations.
                                        Lower = more aggressive (1 = complete if ≥1% in extract).
                                        Set to 0 to disable. Default 1 (most complete).
        """
        self.planet_path = Path(planet_path)
        self.cache_dir = Path(cache_dir)
        self.grid_size = grid_size
        self.complete_partial_relations = complete_partial_relations

        # Create cache directories
        self.extracts_dir = self.cache_dir / "extracts"
        self.extracts_dir.mkdir(parents=True, exist_ok=True)

    def _bbox_key(self, bbox: tuple[float, float, float, float]) -> str:
        """Generate cache key for a bbox (after snapping to grid)."""
        snapped = snap_bbox_to_grid(bbox, self.grid_size)
        return bbox_to_cache_key(snapped)

    def select_source(self, bbox: tuple[float, float, float, float]) -> Path:
        """
        Find smallest extract that fully contains bbox.

        Searches extracts_dir for bbox-named PBF files that contain
        the requested bbox, returning the smallest one. Falls back
        to planet_path if no suitable extract exists.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees

        Returns:
            Path to the best source file (smallest containing extract or planet)
        """
        candidates: list[tuple[int, Path, tuple[float, float, float, float]]] = []

        for pbf in self.extracts_dir.glob("bbox_*.osm.pbf"):
            # Skip temp and lock files
            if ".tmp." in pbf.name or pbf.name.endswith(".lock"):
                continue

            extract_bbox = parse_bbox_from_filename(pbf.name)
            if extract_bbox is None:
                continue

            if bbox_contains(extract_bbox, bbox):
                size = pbf.stat().st_size
                # Skip empty/corrupt files (valid PBF is at least a few KB)
                if size < 1000:
                    continue
                candidates.append((size, pbf, extract_bbox))

        if candidates:
            # Return smallest file that contains our bbox
            candidates.sort(key=lambda x: x[0])
            selected = candidates[0]
            logger.info(
                "Selected source: %s (%d bytes) for bbox %s",
                selected[1].name,
                selected[0],
                bbox,
            )
            return selected[1]

        # Fall back to planet
        logger.info("No regional extract found for bbox %s, using planet file", bbox)
        return self.planet_path

    def get_extract(
        self,
        bbox: tuple[float, float, float, float],
        snap_to_grid: bool = True,
    ) -> Path:
        """
        Get PBF extract for bbox, creating if not cached.

        Automatically selects the smallest available source extract
        that contains the requested bbox.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            snap_to_grid: Whether to snap bbox to grid for cache lookup (default True)

        Returns:
            Path to the PBF extract file

        Raises:
            OsmCacheError: If extraction fails
            FileNotFoundError: If planet file doesn't exist
        """
        # Snap to grid for caching
        cache_bbox = snap_bbox_to_grid(bbox, self.grid_size) if snap_to_grid else bbox
        key = bbox_to_cache_key(cache_bbox)
        cached_path = self.extracts_dir / f"{key}.osm.pbf"
        lock_path = self.extracts_dir / f"{key}.lock"

        # Fast path: already cached
        if cached_path.exists():
            return cached_path

        # Select best source (smallest extract containing bbox, or planet)
        source = self.select_source(cache_bbox)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Slow path: need to extract with locking
        return self._extract_with_lock(cache_bbox, cached_path, lock_path, source)

    def _extract_with_lock(
        self,
        bbox: tuple[float, float, float, float],
        output_path: Path,
        lock_path: Path,
        source: Path,
    ) -> Path:
        """
        Extract bbox from source file with file-based locking.

        Uses fcntl for POSIX file locking to prevent concurrent workers
        from extracting the same bbox simultaneously.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            output_path: Where to write the extracted PBF
            lock_path: Lock file path for concurrency control
            source: Source PBF file to extract from
        """
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                # Double-check after acquiring lock (another worker may have finished)
                if output_path.exists():
                    return output_path

                self._run_osmium_extract(bbox, output_path, source)
                return output_path
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _run_osmium_extract(
        self,
        bbox: tuple[float, float, float, float],
        output_path: Path,
        source: Path,
    ) -> None:
        """
        Run osmium extract command.

        Args:
            bbox: (west, south, east, north) in WGS84 degrees
            output_path: Where to write the extracted PBF
            source: Source PBF file to extract from

        Raises:
            OsmCacheError: If osmium command fails
        """
        west, south, east, north = bbox
        bbox_str = f"{west},{south},{east},{north}"

        # Use temp file to avoid partial writes on failure
        temp_path = output_path.with_suffix(".tmp.osm.pbf")

        cmd = [
            "osmium",
            "extract",
            "--bbox",
            bbox_str,
            "--strategy",
            "smart",
            "-S",
            "types=multipolygon",
        ]

        # Add complete-partial-relations if enabled
        if self.complete_partial_relations > 0:
            cmd.extend(["-S", f"complete-partial-relations={self.complete_partial_relations}"])

        cmd.extend([
            str(source),
            "-o",
            str(temp_path),
            "--overwrite",
        ])

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            # Atomic rename on success
            temp_path.rename(output_path)
        except subprocess.CalledProcessError as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise OsmCacheError(
                f"osmium extract failed: {e.stderr or e.stdout or str(e)}"
            ) from e
        except FileNotFoundError as e:
            raise OsmCacheError(
                "osmium command not found. Install with: apt install osmium-tool"
            ) from e

    def get_cache_info(self) -> dict[str, int | list[str]]:
        """
        Get information about cached extracts.

        Returns:
            Dict with count and list of cached bbox keys
        """
        cached_files = list(self.extracts_dir.glob("bbox_*.osm.pbf"))
        return {
            "count": len(cached_files),
            "keys": [f.stem for f in cached_files],
            "total_size_mb": sum(f.stat().st_size for f in cached_files) // (1024 * 1024),
        }

    def clear_cache(self) -> int:
        """
        Remove all cached extracts.

        Returns:
            Number of files removed
        """
        count = 0
        for f in self.extracts_dir.glob("bbox_*"):
            f.unlink()
            count += 1
        return count
