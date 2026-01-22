from dataclasses import dataclass
from typing import Dict, Tuple, Union, Sequence

import numpy as np
import pymap3d as pm
from ellipticipy import ellipticity_correction
from obspy import Stream, Trace, UTCDateTime


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


def trim_waveform(
    data: Union[Stream, Trace],
    origin_time: UTCDateTime,
    start_offset: Union[float, int, Sequence[Union[float, int]]],
    end_offset: Union[float, int, Sequence[Union[float, int]]],
    nearest_sample: bool = True,
    pad: bool = False,
    fill_value: Union[int, float, None] = None
) -> Union[Stream, Trace]:
    """
    Trim seismic waveform data (Trace or Stream) relative to an earthquake origin time.

    This function allows users to specify trimming window using relative time offsets
    (in seconds) from the origin time.
    The original metadata and structure of the input data are preserved in the output.

    Args:
        data: Input waveform data, either a Trace or a Stream object.
        origin_time: Earthquake origin time (UTCDateTime) as reference point.
        start_offset: Start time relative to origin_time, in seconds (e.g., -5.0 for 5s before).
        end_offset: End time relative to origin_time, in seconds (e.g., 30.0 for 30s after).
        nearest_sample: Passed to ObsPy trim() — whether to allow sample interpolation.
        pad: Passed to ObsPy trim() — whether to pad if window exceeds data bounds.
        fill_value: Value to use for padding if 'pad=True'.

    Returns:
        Trimmed waveform data of the same type as input (Trace or Stream).
        All metadata and structure are preserved.

    Raises:
        ValueError:
        1. If start_offset >= end_offset.
        2. Trim with unexpected failure
        TypeError:
        1. If input data is neither Trace nor Stream.
        2. start_offset or end_offset must format like:
            Union[float, int, Sequence[Union[float, int]]]
        3. For Trace, offsets are more than one number
    """
    # Handle single Trace
    # only scalar offsets allowed
    if isinstance(data, Trace):
        if (not isinstance(start_offset, (int, float)) or
                not isinstance(end_offset, (int, float))):
            raise TypeError("For Trace, offsets must be single numbers.")
        if start_offset >= end_offset:
            raise ValueError("start_offset must be < end_offset.")

        t1 = origin_time + start_offset
        t2 = origin_time + end_offset
        return data.copy().trim(starttime=t1, endtime=t2,
                                nearest_sample=nearest_sample,
                                pad=pad, fill_value=fill_value)

    # Handle Stream
    # support scalar or sequence offsets
    if isinstance(data, Stream):
        n = len(data)

        # Normalize offsets at length n
        def _normalize_offset(offset, name):
            if isinstance(offset, (int, float)):
                return [offset] * n
            elif isinstance(offset, (list, tuple, np.ndarray)):
                if len(offset) != n:
                    raise ValueError(f"{name} length ({len(offset)}) != number of traces ({n}).")
                return list(offset)
            else:
                raise TypeError(f"{name} must be number or sequence of numbers.")

        starts = _normalize_offset(start_offset, "start_offset")
        ends   = _normalize_offset(end_offset,   "end_offset")

        # Validate each window
        for i, (start, end) in enumerate(zip(starts, ends)):
            if start >= end:
                tr_id = data[i].id
                raise ValueError(f"Invalid window for trace {i} ({tr_id}): "
                                 f"start_offset={start} >= end_offset={end}")

        # Trim each trace
        out_stream = Stream()
        for i, tr in enumerate(data):
            try:
                t1 = origin_time + starts[i]
                t2 = origin_time + ends[i]
                trimmed = tr.copy().trim(starttime=t1, endtime=t2,
                                         nearest_sample=nearest_sample,
                                         pad=pad, fill_value=fill_value)
                out_stream += trimmed
            except Exception as e:
                raise ValueError(f"Trim failed for trace {tr.id} (index {i})") from e

        return out_stream

    # Invalid input type
    raise TypeError("Input must be Trace or Stream.")


def cut_signal_win(st: Stream, event_info: SeismicEvent, win=(1, 2)) -> Tuple[Stream, Stream]:
    ref_pkikp = np.array([tr.stats.sac['t0'] for tr in st])
    ref_pcp = np.array([tr.stats.sac['t1'] for tr in st])

    start_pkikp = ref_pkikp - win[0]
    end_pkikp = ref_pkikp + win[1]
    signal_pkikp = trim_waveform(st, event_info.origntime, start_pkikp, end_pkikp)

    start_pcp = ref_pcp - win[0]
    end_pcp = ref_pcp + win[1]
    signal_pcp = trim_waveform(st, event_info.origntime, start_pcp, end_pcp)

    return signal_pcp, signal_pkikp    # first is PcP slice, second is PKiKP


def get_core_phase_amp(st: Stream, event_info: SeismicEvent) -> Dict[str, float]:
    """Compute PKiKP/PcP amplitude ratio for each station."""
    signal_pcp, signal_pkikp = cut_signal_win(st, event_info)

    # Ensure paired traces (critical fix)
    assert len(signal_pkikp) == len(signal_pcp), "PKiKP and PcP trace counts mismatch"

    amplitude = {}
    for tr_pcp, tr_pkikp in zip(signal_pcp, signal_pkikp):
        if tr_pcp.id != tr_pkikp.id:
            print(f"Mismatch PcP: {tr_pcp.id} and PKiKP: {tr_pcp.id}")
            continue

        sta = f"{tr_pcp.stats.network}.{tr_pcp.stats.station}"

        amp_pkikp = np.max(np.abs(tr_pkikp.data))
        amp_pcp = np.max(np.abs(tr_pcp.data))

        amplitude[sta] = np.log10(amp_pkikp / amp_pcp) if amp_pcp > 0 else np.nan

    return amplitude