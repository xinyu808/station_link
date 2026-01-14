from obspy import Stream
from typing import Dict, List
import numpy as np
from scipy.spatial import cKDTree

from .utils import geo_coord2xyz, get_core_phase_time
from .model import StationPoint, StationLink, StationInfo, SeismicEvent


def get_station_point(
        st: Stream,
        station_info_dict: Dict[str, StationInfo],
        event_info: SeismicEvent
) -> List[StationPoint]:
    """
    Construct a list of StationPoint objects by combining static station metadata
    with travel-time residuals for a given seismic event.

    Parameters
    ----------
    st : Stream
        ObsPy stream containing waveform data with SAC headers (e.g., t0, t1, gcarc).
    station_info_dict : Dict[str, StationInfo]
        Dictionary mapping station IDs (e.g., "NET.STA") to their geodetic metadata.
    event_info : SeismicEvent
        Event metadata including origin time, location, depth, and event ID.

    Returns
    -------
    List[StationPoint]
        A list of validated StationPoint instances, each containing station ID,
        event ID, ECEF coordinates (km), and a finite PKiKP travel-time residual.
    """
    stations_point = []

    # Convert geodetic coordinates
    static_info_dict = geo_coord2xyz(station_info_dict)

    # event id: year-month-day
    ev_id = event_info.origntime.strftime("%Y-%m-%d")

    # Compute travel-time residuals
    residual_dict = get_core_phase_time(st, event_info)

    # Merge to make up StationPoint class
    import numpy as np

    for sta_id, static_base in static_info_dict.items():

        res_val = residual_dict.get(sta_id)
        if res_val is None or np.isnan(res_val):
            continue

        stations_point.append(
            StationPoint(
                sta_name=sta_id,
                ev_id=ev_id,
                xyz=static_base.xyz,
                residual=res_val
            )
        )

    return stations_point


def get_station_pairs(
        points_a: List[StationPoint],
        points_b: List[StationPoint],
        max_radius_km: float
) -> List[StationLink]:
    """
    Matches stations between two sets using spatial proximity (KD-Tree).

    This function establishes geographically proximal pairs between reference (points_a)
    and target (points_b) station sets. For each station in points_a, finds the closest
    station in points_b within max_radius_km. Handles both identical station names (distance≈0)
    and spatial neighbors (distance>0) automatically.

    Key features:
    - Uses KD-Tree for efficient O(n log n) nearest neighbor search
    - Automatically prioritizes exact station matches (distance=0) over spatial neighbors
    - Filters out pairs exceeding max_radius_km
    - Preserves all original station metadata in StationLink objects

    Parameters
    ----------
    points_a : List[StationPoint]
        Reference station set (e.g., event1 stations).
        Will be matched against points_b.
    points_b : List[StationPoint]
        Target station set (e.g., event2 stations).
        Served as the search pool for points_a.
    max_radius_km : float
        Maximum search radius (km). Pairs beyond this distance are discarded.
        Typical value: 500-1000 km (adjust based on seismic network density).

    Returns
    -------
    List[StationLink]
        List of matched station pairs. Each StationLink contains:
        - point_a: Reference station (from points_a)
        - point_b: Target station (from points_b)
        - distance_km: Actual geodesic distance (km) between stations
        - link_type: "NEIGHBOR" (always for this implementation)
    """
    # Early exit for empty inputs
    if not points_a or not points_b:
        return []

    # Build KD-Tree from points_b coordinates
    coords_b = np.array([p.xyz for p in points_b])
    tree = cKDTree(coords_b)

    links = []

    # Search for nearest neighbors in points_b for each point in points_a
    for p_a in points_a:
        dist, idx = tree.query(
            p_a.xyz,
            k=1,
            distance_upper_bound=max_radius_km
        )

        # Skip if no valid match found within radius
        if dist == float('inf'):
            continue

        # Retrieve matched station from points_b
        p_b = points_b[idx]

        # Create StationLink object with distance
        links.append(StationLink(
            point_a=p_a,
            point_b=p_b,
            distance_km=float(dist)
        ))

    return links