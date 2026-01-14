from typing import Tuple, Literal
from dataclasses import dataclass

from geographiclib.geodesic import Geodesic
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel

@dataclass(frozen=True)
class StationPoint:
   """
   代表一个真实物理观测点
   """
   sta_name: str          # 台站名
   ev_id: str           # 事件标签
   xyz: Tuple[float, float, float]      # XYZ坐标系统
   residual: float           # PKiKP — PcP 观测值与理论值差


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
            [self.point_a.sta_name, self.point_b.sta_name]
        )
        return f"{names[0]}-{names[1]}"

    @property
    def link_type(self) -> Literal["SAME", "NEIGHBOR"]:
        """
        判断链接类型
        """
        a_sta_name = self.point_a.sta_name
        b_sta_name = self.point_b.sta_name

        if a_sta_name == b_sta_name:
            return "SAME"
        return "NEIGHBOR"

    @property
    def res_diff(self) -> float:
        return self.point_a.residual - self.point_b.residual

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

    def calculate_distance_to(self, event: SeismicEvent):
        """
        Calculate the great-circle distance (gcarc) and azimuth (az)
        from the station to the specified earthquake event.
        """
        dist_m, az, baz = gps2dist_azimuth(
            event.latitude, event.longitude,
            self.latitude, self.longitude
        )
        gcarc = kilometers2degrees(dist_m / 1000.0)
        return gcarc, az

class SeismicModel:
    MODEL_NAME = "prem"
    MODEL = TauPyModel(MODEL_NAME)
    GEOD = Geodesic.WGS84

    @staticmethod
    def get_velocity_model():
        """Get the velocity config from TauPyModel"""
        return SeismicModel.MODEL.model.s_mod.v_mod