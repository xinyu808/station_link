from obspy import Stream
from typing import Dict, List
import numpy as np
from scipy.spatial import cKDTree

from .utils import geo_coord2xyz, get_core_phase_time, get_core_phase_amp
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
    amplitude_dict = get_core_phase_amp(st, event_info)

    # Merge to make up StationPoint class
    import numpy as np

    for sta_id, static_base in static_info_dict.items():

        res_val = residual_dict.get(sta_id)
        amp_val = amplitude_dict.get(sta_id)

        if res_val is None or np.isnan(res_val):
            continue
        if amp_val is None or np.isnan(amp_val):
            continue

        stations_point.append(
            StationPoint(
                station_id=sta_id,
                ev_id=ev_id,
                latitude=static_base.lat,
                longitude=static_base.lon,
                elevation=static_base.elev,
                xyz=static_base.xyz,
                residual=res_val,
                amplitude=amp_val
            )
        )

    return stations_point


def get_station_pairs(
        points_a: List[StationPoint],
        points_b: List[StationPoint],
        max_radius_km: float
) -> List[StationLink]:
    """
    Establish symmetric station pairs between two sets using bi-directional KD-Tree search.

    This function performs both A->B and B->A nearest neighbor searches to maximize
    the number of geographic links, especially useful when seismic networks differ
    between events. It ensures symmetry while maintaining the identity of Event A
    and Event B: 'point_a' always originates from 'points_a'.

    Key features:
    - Bi-directional Search: Captures all proximal pairs within max_radius_km.
    - Identity Preservation: Fixed slotting ensures consistent (A - B) residual math.
    - Deduplication: Uses (sta_a, sta_b) keys to handle mutual nearest neighbors.
    - O(n log n) Efficiency: Leverages cKDTree for rapid spatial querying.

    Parameters
    ----------
    points_a : List[StationPoint]
        Station set from the first event (Reference).
        Populates 'point_a' in the resulting StationLink.
    points_b : List[StationPoint]
        Station set from the second event (Target).
        Populates 'point_b' in the resulting StationLink.
    max_radius_km : float
        Maximum spatial distance (km) for matching. Pairs exceeding this are ignored.

    Returns
    -------
    List[StationLink]
        List of unique station pairs. Each StationLink contains:
        - point_a: Station from points_a.
        - point_b: Station from points_b.
        - distance_km: Precise ECEF distance (km) between the pair.
        - link_type: "SAME" or "NEIGHBOR" based on station IDs.
    """
    if not points_a or not points_b:
        return []

    tree_a = cKDTree(np.array([p.xyz for p in points_a]))
    tree_b = cKDTree(np.array([p.xyz for p in points_b]))

    unique_links: Dict[tuple, StationLink] = {}

    def search_and_add(source_list, target_tree, target_list, is_reverse: bool):
        for p_src in source_list:
            dist, idx = target_tree.query(
                p_src.xyz,
                k=1,
                distance_upper_bound=max_radius_km
            )
            if dist == float('inf'):
                continue

            p_tgt = target_list[idx]
            pa, pb = (p_tgt, p_src) if is_reverse else (p_src, p_tgt)
            link_key = (pa.sta_name, pb.sta_name)

            if link_key not in unique_links:
                unique_links[link_key] = StationLink(
                    point_a=pa,
                    point_b=pb,
                    distance_km=float(dist)
                )

    # Forward: A → B
    search_and_add(points_a, tree_b, points_b, is_reverse=False)
    # Reverse: B → A (captures additional pairs)
    search_and_add(points_b, tree_a, points_a, is_reverse=True)

    return list(unique_links.values())