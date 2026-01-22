from typing import Tuple, Literal, Optional
from dataclasses import dataclass, field

import numpy as np
from geographiclib.geodesic import Geodesic
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel


# ----- config -----
@dataclass
class SeismicEvent:
    """Seismic Event Info"""
    event_description: str
    origntime: UTCDateTime
    latitude: float
    longitude: float
    depth: float  # km
    magnitude: float
    tensor: list


@dataclass(frozen=True)
class StationInfo:
    """Station Info"""
    station_id: str
    network: str
    station: str
    latitude: float
    longitude: float
    elevation: float  # km


class SeismicModel:
    MODEL_NAME = "prem"
    MODEL = TauPyModel(MODEL_NAME)
    GEOD = Geodesic.WGS84

    @staticmethod
    def get_velocity_model():
        """Get the velocity config from TauPyModel"""
        return SeismicModel.MODEL.model.s_mod.v_mod

# -- package core class
@dataclass(frozen=True)
class StationPoint:
    """
    Spatial coordinates of sampling points,
    along with amplitudes and travel times.
    """
    # id tag
    station_id: str  # f"{network}.{station}"
    ev_id: str  # f"{year}-{month}-{day}"

    # coordinate
    latitude: float
    longitude: float
    elevation: float  # km
    xyz: Tuple[float, float, float]  # XYZ for KD-Tree

    # waveform information
    residual: Optional[float] = field(default=None, compare=False, hash=False)
    # amp: log10[PKiKP/PcP]
    amplitude: Optional[float] = field(default=None, compare=False, hash=False)


    def calc_geo_path(self, event_info: SeismicEvent):
        """
        Calc the Epicenter distance (gcarc) and azimuth (az)
        from the station to the specified earthquake event.
        """
        dist_m, az, baz = gps2dist_azimuth(
            event_info.latitude, event_info.longitude,
            self.latitude, self.longitude
        )
        gcarc = kilometers2degrees(dist_m / 1000.0)

        return gcarc, az


    def bounce_point(self, event_info: SeismicEvent, phase="PKiKP"):
        """
        calc bounce point
        """
        # 1. calc gcarc
        gcarc, azi = self.calc_geo_path(event_info)

        # 2. calc ray path
        arrivals = SeismicModel.MODEL.get_ray_paths(
            source_depth_in_km=event_info.depth,
            distance_in_degree=gcarc,
            phase_list=[phase]
        )

        if not arrivals:
            return None, None

        # 3. calc the deepest
        path = arrivals[0].path
        depths = path['depth']
        distances_rad = path['dist']

        # calc bounce points
        idx = np.argmax(depths)
        refl_dist_deg = np.degrees(distances_rad[idx])

        # 4. projection
        distance_m = refl_dist_deg * 111194.9  # 1≈111km
        direct = SeismicModel.GEOD.Direct(
            event_info.latitude,
            event_info.longitude,
            azi, distance_m
        )
        proj_lat = direct['lat2']
        proj_lon = direct['lon2']

        return proj_lat, proj_lon


@dataclass(frozen=True)
class StationLink:
    """
    代表两个观测点之间的连接
    """
    point_a: StationPoint
    point_b: StationPoint
    distance_km: float

    @property
    def id(self) -> str:
        """生成唯一的 ID (A-B 和 B-A 相同)"""
        names = sorted(
            [self.point_a.station_id, self.point_b.station_id]
        )
        return f"{names[0]}-{names[1]}"

    @property
    def link_type(self) -> Literal["SAME", "NEIGHBOR"]:
        """
        判断链接类型
        """
        a_sta_name = self.point_a.station_id
        b_sta_name = self.point_b.station_id

        if a_sta_name == b_sta_name:
            return "SAME"
        return "NEIGHBOR"

    @property
    def res_diff(self) -> float:
        return self.point_a.residual - self.point_b.residual

    @property
    def amp_diff(self) -> float:
        return self.point_a.amplitude - self.point_b.amplitude