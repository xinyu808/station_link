# 寻找邻近台

通过双向映射 **KD-Tree**，在不同地震事件之间寻找“相同台站”或“邻近台站”，从而实现数据增广。

## 1. 功能与目的

* **数据标准化**：将 **经纬度坐标** 转换为 **直角坐标**，便于寻找邻近。
* **残差计算**： 自动计算观测到时与理论到时（PREM模型 + 椭率校正）的差分残差 $(PKiKP - PcP)_{obs} - (PKiKP - PcP)_{ref}$。
* **动态配对**：利用 KD-Tree 算法，在两个地震事件的台站集合中寻找最佳匹配。
    * 优先匹配**同名台站**（距离 ≈ 0）。
    * 缺失同名台站时，自动寻找**邻近台站**（距离 < 阈值）作为物理替补。

---

## 2. 快速开始 (Quick Start)

### 安装依赖
确保您的环境已安装以下依赖库：
```bash
pip install numpy scipy obspy pymap3d geographiclib ellipticipy
```

### 最小示例代码
```python
from obspy import read
from link_package import (
    StationInfo, SeismicEvent,
    get_station_point, get_station_pairs
)

# 1. 准备静态台站信息 (通常来自 Inventory 或 XML)
# 格式: Dict[station_id, StationInfo]
station_inventory = {
    "NET.STA1": StationInfo(station_id="NET.STA1", network="NET", station="STA1", 
                            latitude=35.0, longitude=135.0, elevation=0.1),
    "NET.STA2": StationInfo(station_id="NET.STA2", network="NET", station="STA2", 
                            latitude=35.05, longitude=135.05, elevation=0.1)
}

# 2. 准备地震事件元数据
event_a_info = SeismicEvent(
    event_description="Event A", origntime=..., latitude=10.0, longitude=120.0, 
    depth=500.0, magnitude=6.0, tensor=[]
)
event_b_info = SeismicEvent(
    event_description="Event B", origntime=..., latitude=10.1, longitude=120.1, 
    depth=510.0, magnitude=6.1, tensor=[]
)

# 3. 读取波形数据 (需包含 t0, t1 SAC头段)
st_a = read("./data/event_a/*.SAC")
st_b = read("./data/event_b/*.SAC")

# 4. 生成单事件观测点集合 (List[StationPoint])
# 这一步会自动进行坐标转换、TauP计算和残差提取
points_a = get_station_point(st_a, station_inventory, event_a_info)
points_b = get_station_point(st_b, station_inventory, event_b_info)

# 5. 核心：跨事件台站匹配
# 寻找半径 50km 内的最佳邻居
links = get_station_pairs(points_a, points_b, max_radius_km=50.0)

# 6. 结果分析
for link in links:
    print(f"Pair: {link.id}")
    print(f"Type: {link.link_type}") # SAME (同台) 或 NEIGHBOR (邻近)
    print(f"Dist: {link.distance_km:.2f} km")
    print(f"Diff: {link.res_diff:.4f} s") # 双差残差
    print("-" * 20)
```

## 3. 数据格式说明
### 3.1 前置输入：波形数据 (ObsPy Stream)
输入的 SAC 文件必须包含手动拾取的震相到时信息：
* **t0**: 观测震相 **PKiKP** 的观测到时。
* **t1**: 参考震相 **PcP** 的观测到时。

注意：程序会读取 tr.stats.sac.get("t0") 和 tr.stats.sac.get("t1") 来计算观测走时差。

### 3.2 中间层：数据模型 (Data Models)
本工具包内部使用 dataclass 保证数据流转的类型安全和逻辑清晰：

---
- 输入模型：StationInfo (静态台站元数据)
> 用于存储从台网（如 Hi-net）获取的原始地理信息
```python
@dataclass(frozen=True)
class StationInfo:
    station_id: str  # 唯一标识符，例如 "NET.STA"
    latitude: float  # 纬度
    longitude: float # 经度
    elevation: float # 高程 (单位: km)
    # ... 其他元数据
```
---
- 核心模型：StationPoint (处理后的观测点)
> get_station_point 函数的输出结果，对经纬度坐标进行了整合：
```python
@dataclass(frozen=True)
class StationPoint:
    sta_name: str                   # 台站名
    ev_id: str                      # 事件ID (通常为日期格式 %Y-%m-%d)
    xyz: Tuple[float, float, float] # 高精度 ECEF 直角坐标 (单位: km)
    residual: float                 # 计算好的走时残差 (Obs - Ref)
```
---
- 结果模型：StationLink (最终配对链接)
> 这是 get_station_pairs 函数生成的最终对象，代表了两个地震事件之间台站的关联关系
```python
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
```