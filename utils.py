from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pymap3d as pm
from ellipticipy import ellipticity_correction
from obspy import Stream, Trace

from .model import StationInfo, SeismicModel, SeismicEvent


@dataclass(frozen=True)
class StationPointBase:
    """Static station metadata with ECEF coordinates (km)."""
    station_id: str
    xyz: Tuple[float, float, float]  # km
    lat: float
    lon: float
    elev: float  # km


def geo_coord2xyz(
    station_info_dict: Dict[str, StationInfo]
) -> Dict[str, StationPointBase]:
    """Convert geodetic coords to ECEF (WGS84), in kilometers."""
    processed_stations: Dict[str, StationPointBase] = {}

    for sta_id, info in station_info_dict.items():
        # Convert elevation to meters for pymap3d
        x_m, y_m, z_m = pm.geodetic2ecef(
            info.latitude,
            info.longitude,
            info.elevation * 1000.0
        )

        # Store ECEF in km
        processed_stations[sta_id] = StationPointBase(
            station_id=sta_id,
            xyz=(x_m / 1000.0, y_m / 1000.0, z_m / 1000.0),
            lat=info.latitude,
            lon=info.longitude,
            elev=info.elevation
        )

    return processed_stations


def calc_ref_tau(tr: Trace, event_info: SeismicEvent) -> Tuple[float, float]:
    """Compute reference PKiKP and PcP arrival times with ellipticity correction."""
    sta = f"{tr.stats.station}"
    stla = tr.stats.coordinates['latitude']
    stlo = tr.stats.coordinates['longitude']
    geod = SeismicModel.GEOD

    # Get theoretical arrivals
    arrivals_PcP = SeismicModel.MODEL.get_ray_paths(
        source_depth_in_km=event_info.depth,
        distance_in_degree=tr.stats.gcarc,
        phase_list=["PcP"]
    )

    arrivals_PKiKP = SeismicModel.MODEL.get_ray_paths(
        source_depth_in_km=event_info.depth,
        distance_in_degree=tr.stats.gcarc,
        phase_list=["PKiKP"]
    )

    # Skip if either phase is missing
    if not arrivals_PKiKP:
        print(f"Warning: No PKiKP at {tr.stats.gcarc}° for station {sta}. Skipping.")
        return np.nan, np.nan

    if not arrivals_PcP:
        print(f"Warning: No PcP at {tr.stats.gcarc}° for station {sta}. Skipping.")
        return np.nan, np.nan

    # Compute azimuth for ellipticity correction
    inv = geod.Inverse(event_info.latitude, event_info.longitude, stla, stlo)
    azi = (inv['azi1'] + 360) % 360

    # Apply ellipticity correction
    correct_PcP = ellipticity_correction(
        arrivals_PcP,
        azimuth=azi,
        source_latitude=event_info.latitude
    )[0]
    correct_PKiKP = ellipticity_correction(
        arrivals_PKiKP,
        azimuth=azi,
        source_latitude=event_info.latitude
    )[0]

    ref_PKiKP = arrivals_PKiKP[0].time + correct_PKiKP
    ref_PcP = arrivals_PcP[0].time + correct_PcP

    return ref_PcP, ref_PKiKP


def get_core_phase_time(st: Stream, event_info: SeismicEvent) -> Dict[str, float]:
    """Compute PKiKP–PcP travel-time residuals for all traces."""
    residual = {}

    for tr in st:
        sta_id = f"{tr.stats.network}.{tr.stats.station}"

        # Get reference (theoretical) differential time
        ref_PcP, ref_PKiKP = calc_ref_tau(tr, event_info)
        if np.isnan(ref_PcP) or np.isnan(ref_PKiKP):
            continue
        ref_diff = ref_PKiKP - ref_PcP

        # Get observed differential time from SAC headers
        obs_PcP = tr.stats.sac.get("t1")
        obs_PKiKP = tr.stats.sac.get("t0")
        if obs_PcP is None or obs_PKiKP is None:
            continue
        obs_diff = obs_PKiKP - obs_PcP

        # Residual = observed - predicted
        residual[sta_id] = obs_diff - ref_diff

    return residual