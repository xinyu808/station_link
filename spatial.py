import io
from typing import List, Dict, Optional, Set, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt

from .model import StationLink, StationPoint, SeismicEvent


def plot_link1d(
    links: List[StationLink],
    x_getter: Callable[[StationLink], float],
    y_getter: Callable[[StationLink], float],
    threshold: float = 1.5,
    show_labels: bool = False,
    xlabel: str = "X Attribute",
    ylabel: str = "Y Attribute",
    title: str = "1D Link Fluctuation Analysis"
):
    """
    Generic 1D scatter plot for analyzing fluctuations (e.g., amplitude or travel-time residuals)
    across spatial or geometric attributes of station-event links.

    Parameters:
    -----------
    links : List[StationLink]
        List of station-event link objects to analyze.

    x_getter : Callable[[StationLink], float]
        Function to extract the x-coordinate
        (e.g., gcarc, azimuth, latitude) from a link.

    y_getter : Callable[[StationLink], float]
        Function to extract the y-value
        (e.g., amplitude, residual) from a link.

    threshold : float, optional
        Multiplier of standard deviation for outlier detection.
        Default is 1.5.

    show_labels : bool, optional
        Whether to annotate outlier points with their link IDs.
        Default is False.

    xlabel : str, optional
        Label for the x-axis.
        Default is "X Attribute".

    ylabel : str, optional
        Label for the y-axis.
        Default is "Y Attribute".

    title : str, optional
        Plot title. Default is "1D Link Fluctuation Analysis".
    """

    if not links:
        print("No links to plot.")
        return


    # -- 1. parse plot data --
    data = []
    for link in links:
        try:
            data.append({
                'x': x_getter(link),
                'y': y_getter(link),
                'type': link.link_type,
                'id': link.point_a.station_id if link.link_type == "SAME" else
                link.id
            })
        except Exception as e:
            continue


    # -- 2. sort and trans to array --
    data.sort(key=lambda d: d['x'])
    x_vals = np.array([d['x'] for d in data])
    y_vals = np.array([d['y'] for d in data])


    # -- 3. calc scatter information --
    mean_y = np.mean(y_vals)
    y_vals = y_vals - mean_y    #demean
    std_y = np.std(y_vals)
    abs_threshold = threshold * std_y


    # -- 4. plotting logic --
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    pad = (x_vals.max() - x_vals.min()) * 0.05


    # -- background --
    ax.fill_between(
        [x_vals.min() - pad, x_vals.max() + pad],
        -abs_threshold, abs_threshold,
        color='gray', alpha=0.3,
        label=f'±{threshold}σ', zorder=1
    )
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=3)


    # -- hist --
    ax_hist = ax.twiny()
    ax_hist.hist(
        y_vals, bins='auto',
        orientation='horizontal',
        color='skyblue', alpha=0.5,
        edgecolor='gray', linewidth=0.3,
        zorder=2
    )
    ax_hist.set_xticks([])    # hide tick for hist plot
    ax_hist.spines['top'].set_visible(False)


    # -- scatter --
    for l_type, color, label in [("SAME", "green", "Same"),
                                 ("NEIGHBOR", "red", "Neighbor")]:
        mask = [d['type'] == l_type for d in data]
        if any(mask):
            # mask point
            ax.scatter(x_vals[mask], y_vals[mask], color=color,
                       edgecolors='k', s=45, alpha=0.75, label=label, zorder=6)

            # mark anomaly point
            if show_labels:
                for i, d in enumerate(data):
                    if d['type'] == l_type and abs(y_vals[i]) > abs_threshold:
                        ax.text(d['x'], float(y_vals[i]) + 0.01, d['id'],
                                fontsize=8, ha='center', va='bottom')


    # -- decorate plotting --
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    stats_text = f"Mean: {mean_y:.3f}\nStd($\\sigma$): {std_y:.3f}"
    ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    ax.legend(loc='upper right', frameon=True)
    ax.set_xlim(x_vals.min() - pad,
                x_vals.max() + pad)

    plt.show()


def parse_plot_data(
    links: List[StationLink],
    event_info1: SeismicEvent,
    event_info2: SeismicEvent,
    value_type: str,    # 'res' or 'amp'
    threshold: Optional[float] = 1.5
):
    """
    Prepares plot data from station links and event info.

    Args:
        links (List[StationLink]): List of station-event links.
        event_info1 (SeismicEvent): Event metadata for point A.
        event_info2 (SeismicEvent): Event metadata for point B.
        value_type (str): Data type to extract, either 'res' or 'amp'.
        threshold (Optional[float]): Outlier threshold in std multiples.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            frame_a, frame_b, and differential frame for plotting.
    """

    # pre judge
    if value_type not in ('res', 'amp'):
        raise ValueError(
            f"Invalid 'value_type': {value_type}. Expected 'res' or 'amp'."
        )

    # Base Parsing (Raw Data Set)
    points_a = {link.point_a for link in links}
    points_b = {link.point_b for link in links}

    info_a = parse_sta_data(points_a, event_info1, value_type)
    info_b = parse_sta_data(points_b, event_info2, value_type)

    mid_info = parse_mid_data(links, info_a, info_b)

    valid_ids = get_valid_link(links, mid_info, info_a, info_b, threshold)

    frame_a, frame_b, frame_diff = build_plot_data(
        links, valid_ids, mid_info, info_a, info_b
    )

    return frame_a, frame_b, frame_diff


def parse_sta_data(
    points: Set[StationPoint],
    event_info: SeismicEvent,
    value_type: str    # 'res' or 'amp'
) -> Dict[str, dict]:
    info = {}

    for p in points:
        name = p.station_id

        lat, lon = p.bounce_point(event_info)
        if lat is None or lon is None:
            continue

        value = p.residual if value_type =='res' else p.amplitude
        if value is None or np.isnan(value):
            continue

        info[name] = {
            'station_id': name,
            'lon': lon,
            'lat': lat,
            'value': value
        }
    return info


def parse_mid_data(
    links: List[StationLink],
    info_a: Dict[str, dict],
    info_b: Dict[str, dict]
) -> Dict[str, dict]:
    """
    Computes midpoint and value difference for each link.

    Args:
        links (List[StationLink]): List of station links.
        info_a (Dict[str, dict]): Metadata for stations at point A, keyed by station ID.
        info_b (Dict[str, dict]): Metadata for stations at point B, keyed by station ID.

    Returns:
        Dict[str, dict]: Midpoint info per link ID, including lat/lon of both ends,
                         midpoint coordinates, and value difference (B - A).
    """
    mid_info = {}

    for link in links:
        s_a = link.point_a.station_id
        s_b = link.point_b.station_id

        if s_a not in info_a or s_b not in info_b:
            continue

        p_info_a = info_a[s_a]
        p_info_b = info_b[s_b]

        mid_info[link.id] = {
            'lat': (p_info_a['lat'] + p_info_b['lat']) / 2,
            'lon': (p_info_a['lon'] + p_info_b['lon']) / 2,
            'lat1': p_info_a['lat'], 'lon1': p_info_a['lon'],
            'lat2': p_info_b['lat'], 'lon2': p_info_b['lon'],
            'diff': p_info_b['value'] - p_info_a['value']
        }
    return mid_info


def get_valid_link(
    links: List[StationLink],
    mid_info: Dict[str, dict],
    info_a: Dict[str, dict],
    info_b: Dict[str, dict],
    threshold: Optional[float]
) -> Set[str]:
    """
       Filters link IDs by outlier detection based on value differences.

       Args:
           links (List[StationLink]): List of station links.
           mid_info (Dict[str, dict]): Midpoint and diff data per link ID.
           info_a (Dict[str, dict]): Station data for point A, keyed by station ID.
           info_b (Dict[str, dict]): Station data for point B, keyed by station ID.
           threshold (Optional[float]): Outlier threshold in σ multiples;
                                       if None, all links are kept.

       Returns:
           Set[str]: Set of valid link IDs that pass the outlier check.
       """
    # threshold None for keep all
    if threshold is None:
        return set(mid_info.keys())

    # -- 1. Calc Raw Means --
    vals_a = [v['value'] for v in info_a.values()]
    vals_b = [v['value'] for v in info_b.values()]

    if not vals_a or not vals_b:
        print("Warning: No valid station data found for stats.")
        return set(mid_info.keys())

    mean_a = np.mean(vals_a)
    mean_b = np.mean(vals_b)

    # Regional Bias (μ and σ)
    raw_mu = mean_b - mean_a
    diffs = np.array([v['diff'] for v in mid_info.values()])
    raw_sigma = np.sqrt(np.mean((diffs - raw_mu) ** 2))

    valid_ids = set()

    # -- Header --
    print(f"\n{'=' * 25} OUTLIER CHECK (Threshold: {threshold}σ) {'=' * 25}")
    print(f"Bias(μ): {raw_mu:.3f} | RMS(σ): {raw_sigma:.3f} | Limit: {threshold * raw_sigma:.3f}")
    print(f"{'Link_ID':<14} {'Sta_A':<8} {'Sta_B':<8} {'Val_A*':>8} {'Val_B*':>8} {'Diff':>8} {'Dev':>8}")
    print(f"*Val = Mean-removed value")
    print("-" * 90)

    link_map = {link.id: link for link in links}

    for lid, info in mid_info.items():
        diff_val = info['diff']
        deviation = abs(diff_val - raw_mu)

        if deviation <= threshold * raw_sigma:
            valid_ids.add(lid)
        else:
            link = link_map[lid]
            s_a = link.point_a.station_id
            s_b = link.point_b.station_id

            # Calc mean-removed Res or Amp for display
            p_info_a = info_a[s_a]
            p_info_b = info_b[s_b]

            ano_vals_a = (p_info_a['value'] - mean_a) if p_info_a else np.nan
            ano_vals_b = (p_info_b['value'] - mean_b) if p_info_b else np.nan

            print(
                f"{lid:<14} {s_a:<8} {s_b:<8} "
                f"{ano_vals_a:8.3f} {ano_vals_b:8.3f} "
                f"{diff_val:8.3f} {deviation:8.3f}"
            )

    print(f"{'=' * 85}\n")
    return valid_ids


def build_plot_data(
    links: List[StationLink],
    valid_ids: Set[str],
    mid_info: Dict[str, dict],
    info_a: Dict[str, dict],
    info_b: Dict[str, dict]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Re-calc clean mean & build output DataFrame
    """
    # Fail Fast
    if not valid_ids:
        empty_cols_a = ['station_id', 'lon', 'lat', 'mean_value']
        empty_cols_b = ['station_id', 'lon', 'lat', 'mean_value']
        empty_cols_diff = ['id', 'lon', 'lat', 'lon1', 'lat1', 'lon2', 'lat2', 'diff']
        return (pd.DataFrame(columns=empty_cols_a),
                pd.DataFrame(columns=empty_cols_b),
                pd.DataFrame(columns=empty_cols_diff))

    # Build Whitelist Logic
    valid_sta_a = set()
    valid_sta_b = set()

    for link in links:

        if link.id not in valid_ids:
            continue
        valid_sta_a.add(link.point_a.station_id)
        valid_sta_b.add(link.point_b.station_id)

    clean_vals_a = [info_a[name]['value'] for name in valid_sta_a if name in info_a]
    clean_vals_b = [info_b[name]['value'] for name in valid_sta_b if name in info_b]

    mean_a = np.mean(clean_vals_a) if clean_vals_a else 0
    mean_b = np.mean(clean_vals_b) if clean_vals_b else 0
    clean_mu = mean_b - mean_a

    df_a = pd.DataFrame([
        {
            'station_id': name,
            'lon': info_a[name]['lon'], 'lat': info_a[name]['lat'],
            'mean_value': info_a[name]['value'] - mean_a
        } for name in valid_sta_a
    ])

    df_b = pd.DataFrame([
        {
            'station_id': name,
            'lon': info_b[name]['lon'], 'lat': info_b[name]['lat'],
            'mean_value': info_b[name]['value'] - mean_b
        } for name in valid_sta_b
    ])

    df_diff = pd.DataFrame([
        {
            'id': lid,
            'lon': mid_info[lid]['lon'], 'lat': mid_info[lid]['lat'],
            'lon1': mid_info[lid]['lon1'], 'lat1': mid_info[lid]['lat1'],
            'lon2': mid_info[lid]['lon2'], 'lat2': mid_info[lid]['lat2'],
            'diff': mid_info[lid]['diff'] - clean_mu
        } for lid in valid_ids
    ])

    return df_a, df_b, df_diff


def plot_link2d(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        df_diff: pd.DataFrame,
        map_range: list,
        output_name: str = "spatial.png"
):
    scale = 1.6
    fig = pygmt.Figure()

    pygmt.config(
        FORMAT_GEO_MAP="dddF",
        MAP_FRAME_TYPE="fancy",
        FONT_ANNOT_PRIMARY="6p",
        MAP_FRAME_WIDTH="2p"
    )

    fig.coast(
        region=map_range,
        projection="M10c",
        land="253/245/230",
        water="skyblue",
        shorelines="1/0.5p",
        area_thresh=5000,
        frame=["xa2", "ya2", "WSrt"]
    )

    # Layer 1: Connection Lines (Filtered)
    for _, row in df_diff.iterrows():
        fig.plot(x=[row.lon1, row.lon2], y=[row.lat1, row.lat2], pen="0.4p,gray70")

    # Layer 2: Plot Unique Stations (Filtered)
    for df, color in zip([df_a, df_b], ["purple", "red"]):
        fig.plot(
            x=df[df['mean_value'] > 0].lon,
            y=df[df['mean_value'] > 0].lat,
            size=df[df['mean_value'] > 0]['mean_value'] * scale,
            style="cc", pen=f"0.2p,{color}"
        )
        fig.plot(
            x=df[df['mean_value'] <= 0].lon,
            y=df[df['mean_value'] <= 0].lat,
            size=df[df['mean_value'] <= 0]['mean_value'].abs() * scale,
            style="tc", pen=f"0.2p,{color}"
        )

    # Layer 3: Paired Diff at Midpoints (Filtered)
    fig.plot(
        x=df_diff[df_diff['diff'] > 0].lon,
        y=df_diff[df_diff['diff'] > 0].lat,
        size=df_diff[df_diff['diff'] > 0]['diff'] * scale,
        style="cc", fill="green", pen="0.2p,black"
    )
    fig.plot(
        x=df_diff[df_diff['diff'] <= 0].lon,
        y=df_diff[df_diff['diff'] <= 0].lat,
        size=df_diff[df_diff['diff'] <= 0]['diff'].abs() * scale,
        style="tc", fill="green", pen="0.2p,black"
    )

    # Layer 4: Legend
    legend_note = io.StringIO(f"""
H 8p,Helvetica-Bold Amplitude (log)
G 0.5
S 0.35c c 0.8c - 0.25p,black 0.95c 0.50
G 0.5
S 0.35c c 0.4c - 0.25p,black 0.95c 0.25
G 0.3
S 0.35c t 0.4c - 0.25p,black 0.865c -0.25
G 0.5
S 0.35c t 0.8c - 0.25p,black 0.865c -0.50
""")

    fig.legend(spec=legend_note, position="jTL+w2.2c+o0.1c/0.1c", box="+p0.45p+gwhite@30")

    fig.savefig(output_name)
    fig.show()