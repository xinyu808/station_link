"""
Neighbor Station Analysis Package
=================================

A package for seismic station pairing based on spatial proximity and
travel-time residual analysis.

Core Functionality:
-------------------
- Calculate high-precision ECEF coordinates from geodetic data.
- Compute PKiKP-PcP travel time residuals relative to reference models.
- Dynamically match stations between events using KD-Tree spatial clustering.

"""

# 2. Expose Data Models (The Nouns)
# 用户最常用的数据对象，直接暴露在包的顶层
from .model import (
    StationPoint,
    StationLink,
    StationInfo,
    SeismicEvent,
    SeismicModel
)

# 3. Expose Core Workflows (The Verbs)
# 核心业务逻辑函数
from .core import (
    get_station_point,
    get_station_pairs
)

# 4. Expose Utilities (The Tools)
# 通常 geo_coord2xyz 用户可能也会单独用到，暴露出来很方便
# StationPointBase 虽然是中间对象，但为了类型提示(Type Hinting)最好也暴露
from .utils import (
    geo_coord2xyz,
    StationPointBase
)

# 5. Define Export List
# 这决定了 `from my_package import *` 会导入什么
# 也是 IDE 自动补全的依据
__all__ = [
    # Models
    "StationPoint",
    "StationLink",
    "StationInfo",
    "SeismicEvent",
    "SeismicModel",

    # Core Functions
    "get_station_point",
    "get_station_pairs",

    # Utilities
    "geo_coord2xyz",
    "StationPointBase",
]