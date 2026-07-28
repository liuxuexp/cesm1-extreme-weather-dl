#!/usr/bin/env python3
"""
01_preprocess.py - Data Preprocessing Pipeline

Generates labeled NetCDF files from CESM1 Large Ensemble raw data:
    1. Read TREFHT / Z500 raw data, crop to China region and merge by season
    2. Compute anomalies, apply land-sea mask
    3. Detect extreme temperature days based on 99th percentile threshold
    4. Cluster extreme events into 4 classes using KMeans
    5. Select normal samples, merge into 5-class label dataset
    6. Generate 6-day lagged sequences (t-5 ~ t) for training

Usage:
    python 01_preprocess.py              # Process both winter and summer
    python 01_preprocess.py -s s         # Summer only
    python 01_preprocess.py -s w         # Winter only
    python 01_preprocess.py --raw /path  # Override raw data directory
"""

import os
import sys
import argparse
import copy
import math
import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn import metrics

import config


def get_all_files(path):
    files = []
    for item in sorted(os.listdir(path)):
        cur = os.path.join(path, item)
        if os.path.isdir(cur):
            files.extend(get_all_files(cur))
        else:
            files.append(cur)
    return files


def filter_by_end(files, endstr="19200101-20051231.nc"):
    return [f for f in files if f.endswith(endstr)]


def split_basename(filename, sep="_", idx=4):
    parts = os.path.basename(filename).split(sep, idx)
    return parts[idx] if len(parts) > idx else ""


def land_sea_mask(ds, mask_file, mode="land"):
    lsm = xr.open_dataset(mask_file)["lsm"]
    common_lat = ds.lat.values
    common_lon = ds.lon.values
    lsm_sel = lsm.sel(lat=common_lat, lon=common_lon, method="nearest")
    if mode == "ocean":
        mask = lsm_sel.values == 0
    else:
        mask = lsm_sel.values == 1
    return ds.where(mask)


def combine_season(raw_dir, var_name, nyears, fd, ld, domain, member_ids, season_name=""):
    season_name = season_name or "unknown"
    savefile = os.path.join(config.OUTPUT_DIR, f"combine-{var_name}-{season_name}-1.nc")
    if os.path.isfile(savefile):
        print(f"  [skip] {savefile} exists, loading...")
        return xr.open_dataset(savefile)

    var_dir = os.path.join(raw_dir, var_name.upper())
    files = filter_by_end(get_all_files(var_dir))
    files.sort()
    listn = [split_basename(f, ".", 4) for f in files]

    lon_w, lon_e = domain["lon_west"], domain["lon_east"]
    lat_s, lat_n = domain["lat_south"], domain["lat_north"]

    combined = None
    for idx, f in enumerate(files):
        print(f"  [{idx+1}/{len(files)}] {os.path.basename(f)}")
        data = xr.open_dataset(f)
        lon_range = data.lon[(data.lon >= lon_w) & (data.lon <= lon_e)]
        lat_range = data.lat[(data.lat >= lat_s) & (data.lat <= lat_n)]
        arr = data[var_name.upper()].sel(lon=lon_range, lat=lat_range).values
        shape = arr.shape

        if var_name == "t2m":
            arr = arr.reshape(nyears, 365, shape[1], shape[2])[:, fd:ld+1, :].reshape(-1, shape[1], shape[2])
        else:
            arr = arr[181:-184, :].reshape(nyears, 365, shape[1], shape[2])[:, fd:ld+1, :].reshape(-1, shape[1], shape[2])

        if combined is None:
            combined = [arr]
        else:
            combined = np.concatenate((combined, [arr]))

    print(f"  combined shape: {combined.shape}")

    data0 = xr.open_dataset(files[0])
    lon_range = data0.lon[(data0.lon >= lon_w) & (data0.lon <= lon_e)]
    lat_range = data0.lat[(data0.lat >= lat_s) & (data0.lat <= lat_n)]

    if var_name == "t2m":
        time_vals = data0.TREFHT.time.values.reshape(nyears, 365)[:, fd:ld+1].flatten()
    else:
        time_vals = data0.Z500.time[181:-184].values.reshape(nyears, 365)[:, fd:ld+1].flatten()

    da = xr.DataArray(combined, coords=[member_ids, time_vals, lat_range.values, lon_range.values],
                      dims=["m", "time", "lat", "lon"])
    ds = xr.Dataset({var_name: da})
    ds.to_netcdf(savefile)
    print(f"  saved: {savefile}")
    return xr.open_dataset(savefile)


def compute_anomaly(ds, nyears, fd, ld, window=15):
    mean = ds.t2m.values.mean(axis=0)
    mean = mean.reshape(nyears, -1, ds.sizes["lat"], ds.sizes["lon"])
    smoothed = np.zeros_like(mean)
    half_w = window // 2
    for i in range(nyears):
        for j in range(half_w, mean.shape[1] - half_w):
            smoothed[i, j, :] = mean[i, j-half_w:j+half_w+1, :].sum(axis=0) / window
    smoothed = smoothed.reshape(-1, ds.sizes["lat"], ds.sizes["lon"])

    anomaly = copy.deepcopy(ds)
    for m in range(ds.sizes["m"]):
        anomaly.t2m.values[m, :] = ds.t2m.values[m, :] - smoothed
    return anomaly


def detect_extremes(anomaly_ds, combined_ds, land_mask_ds, pct=99,
                    min_consec=5, min_sep=16, n_clusters=4, n_normal=800):
    data = anomaly_ds.t2m.values
    flat = data.reshape(config.N_MEMBERS * data.shape[0] // config.N_MEMBERS, -1)
    n_members = config.N_MEMBERS
    n_times = data.shape[0]
    n_total = n_members * n_times
    n_lat = data.shape[1]
    n_lon = data.shape[2]

    flat = data.reshape(n_total, n_lat * n_lon)
    p99 = np.nanpercentile(flat, pct, axis=0)
    print(f"  p{pct} shape: {p99.shape}")

    extreme_mask = np.empty(flat.shape, dtype=bool)
    extreme_mask.fill(False)
    for i in range(n_total):
        extreme_mask[i, :] = np.logical_and(flat[i, :] >= p99, flat[i, :] >= 3)
        if i % 1000 == 0:
            print(f"    checking day {i}/{n_total}")

    extreme_mask_simple = np.empty(flat.shape, dtype=bool)
    extreme_mask_simple.fill(False)
    for i in range(n_total):
        extreme_mask_simple[i, :] = flat[i, :] >= p99

    extreme_mask_3d = extreme_mask.reshape(n_members, n_times, n_lat, n_lon)
    extreme_mask_simple_3d = extreme_mask_simple.reshape(n_members, n_times, n_lat, n_lon)

    heatwave = np.empty((n_members, n_times), dtype=bool)
    heatwave.fill(False)
    heatwave_simple = np.empty((n_members, n_times), dtype=bool)
    heatwave_simple.fill(False)
    heatwave_simple2 = np.empty((n_members, n_times), dtype=bool)
    heatwave_simple2.fill(False)

    temp = extreme_mask_simple_3d.reshape(n_members, n_times, -1).swapaxes(1, 2).reshape(-1, n_times)
    temp2 = extreme_mask_3d.reshape(n_members, n_times, -1).swapaxes(1, 2).reshape(-1, n_times)

    for i in range(min_consec, n_times - min_consec - min_sep):
        t_5 = temp2[:, i:i+min_consec].all(axis=1)
        t0 = temp[:, i]
        t02 = temp2[:, i]
        heatwave[:, i] = t_5.reshape(n_members, -1).any(axis=1)
        heatwave_simple[:, i] = t0.reshape(n_members, -1).any(axis=1)
        heatwave_simple2[:, i] = t02.reshape(n_members, -1).any(axis=1)

    print(f"  heatwave count before rule: {heatwave.sum()}")

    htd = copy.deepcopy(heatwave)
    for i in range(htd.shape[0]):
        for j in range(n_times - min_sep, min_consec + min_sep - 1, -1):
            if htd[i, j] and htd[i, j-1]:
                htd[i, j] = False

    onsetlist = []
    onsetlistv1 = []
    onsetlistv2 = []
    for i in range(htd.shape[0]):
        for j in range(min_consec, n_times - min_consec - min_sep):
            if htd[i, j] and not heatwave_simple[i, j-min_sep:j].any() and not htd[i, j+1:j+min_sep].any():
                member_idx = i
                day_idx = j
                onsetlist.append([member_idx // n_times, member_idx % n_times, day_idx])
                onsetlistv1.append([member_idx, day_idx])
                onsetlistv2.append(member_idx * n_times + day_idx)

    print(f"  extreme onset days: {len(onsetlist)}")

    onset_pattern = extreme_mask.reshape(-1, n_lat * n_lon)
    onset_day_pattern = onset_pattern[onsetlistv2]

    t2m_flat = data.reshape(n_total, n_lat * n_lon)
    onset_t2m_flat = t2m_flat[onsetlistv2]

    def pooling(feature_map, size=(4, 4), stride=(2, 2)):
        c, h, w = feature_map.shape
        ph = int(math.ceil(h / stride[0]))
        pw = int(math.ceil(w / stride[1]))
        out = np.zeros((c, ph, pw), dtype=float)
        for m in range(c):
            for r_i, r in enumerate(range(0, h, stride[0])):
                for c_i, c in enumerate(range(0, w, stride[1])):
                    out[m, r_i, c_i] = np.nanmax(feature_map[m, r:r+size[0], c:c+size[1]])
        return out

    clustered_data = pooling(onset_t2m_flat.reshape(-1, n_lat, n_lon), (2, 2), (1, 1))
    clustered_data = clustered_data.reshape(clustered_data.shape[0], -1)
    clustered_data = np.nan_to_num(clustered_data, 0)

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=50, random_state=10, max_iter=50).fit(clustered_data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    print(f"  kmeans inertia: {kmeans.inertia_:.2f}, n_iter: {kmeans.n_iter_}")
    print(f"  silhouette: {metrics.calinski_harabasz_score(clustered_data, labels):.2f}")

    for cl in range(n_clusters):
        print(f"    cluster {cl}: {np.sum(labels == cl)} samples")

    otherdays = np.empty(heatwave_simple2.reshape(-1).shape[0], dtype=bool)
    otherdays.fill(False)
    otherdays[:] = heatwave_simple2.reshape(-1)[:]
    for idx in onsetlistv2:
        otherdays[idx - min_consec:idx + min_sep] = True
    otherdays = otherdays.reshape(n_members, n_times)
    otherdays[:, :min_consec] = True
    otherdays[:, n_times - min_consec:] = True
    otherdays = ~otherdays

    normallist = []
    for i in range(otherdays.shape[0]):
        for j in range(min_consec, n_times - min_consec - min_sep):
            if otherdays[i, j]:
                normallist.append([i // n_times, i % n_times, j])

    print(f"  total normal candidates: {len(normallist)}")

    step = max(1, len(normallist) // n_normal)
    normallist_sampled = normallist[::step][:n_normal]

    normallistv1 = [[item[0] * n_times + item[1], item[2]] for item in normallist_sampled]
    normallistv2 = [(item[0] * n_times + item[1]) * n_times + item[2] for item in normallist_sampled]
    print(f"  normal samples after sampling: {len(normallist_sampled)}")

    normalt2m = t2m_flat[normallistv2]
    normalt2m = normalt2m[:, ~np.isnan(normalt2m).all(axis=0)]

    center0 = np.zeros((1, normalt2m.shape[1]), dtype=float)
    normallabel = np.zeros(normalt2m.shape[0], dtype=int)
    labels2 = np.concatenate((normallabel, labels + 1))
    print(f"  merged labels shape: {labels2.shape}")

    datalist = normallist_sampled + onsetlist
    datalistv2 = normallistv2 + onsetlistv2

    knum = n_clusters + 1
    llist = [np.where(labels2 == i) for i in range(knum)]

    mindl = [0] * knum
    minll = [0] * knum

    mind = 9999
    minl = 0
    for j in llist[0][0]:
        d = np.linalg.norm(normalt2m[j] - center0.squeeze())
        if mind > d:
            mind = d
            minl = j
    minll[0] = minl
    mindl[0] = mind

    for i in range(1, knum):
        mind = 9999
        minl = 0
        for j in llist[i][0]:
            data_idx = j - len(llist[0][0])
            d = np.linalg.norm(clustered_data[data_idx] - centers[i-1])
            if mind > d:
                mind = d
                minl = j
        minll[i] = minl
        mindl[i] = mind

    print(f"  center sample indices: {minll}")

    return labels2, datalistv2, minll, n_clusters, onset_day_pattern


def build_lagged_dataset(combined_t2m, combined_z500, anomaly_ds, labels2, datalistv2, minll, n_clusters, lead_lag=6):
    n_total = combined_t2m.sizes["m"] * combined_t2m.sizes["time"]
    n_lat_t2m = combined_t2m.sizes["lat"]
    n_lon_t2m = combined_t2m.sizes["lon"]
    n_lat_z500 = combined_z500.sizes["lat"]
    n_lon_z500 = combined_z500.sizes["lon"]

    ot2m = combined_t2m.t2m.values.reshape(n_total, n_lat_t2m * n_lon_t2m)
    ot2man = anomaly_ds.t2m.values.reshape(n_total, n_lat_t2m * n_lon_t2m)
    oz500 = combined_z500.z500.values.reshape(n_total, n_lat_z500 * n_lon_z500)

    ol = np.asarray(datalistv2)

    dayt2m = np.concatenate([ot2m[ol - d] for d in range(lead_lag)], axis=1)
    dayt2man = np.concatenate([ot2man[ol - d] for d in range(lead_lag)], axis=1)
    dayz500 = np.concatenate([oz500[ol - d] for d in range(lead_lag)], axis=1)

    dayt2m = dayt2m.reshape(-1, lead_lag, n_lat_t2m, n_lon_t2m)
    dayt2man = dayt2man.reshape(-1, lead_lag, n_lat_t2m, n_lon_t2m)
    dayz500 = dayz500.reshape(-1, lead_lag, n_lat_z500, n_lon_z500)

    num = datalistv2
    day = list(range(lead_lag))
    classnum = list(range(n_clusters + 1))
    centersnum = np.asarray(datalistv2)[minll]
    centers = minll

    lon_t2m = combined_t2m.lon
    lat_t2m = combined_t2m.lat
    lon_z500 = combined_z500.lon
    lat_z500 = combined_z500.lat

    ds_t2m = xr.Dataset({
        "t2m": xr.DataArray(dayt2m, coords=[num, day, lat_t2m, lon_t2m],
                            dims=["num", "day", "lat", "lon"]),
        "t2man": xr.DataArray(dayt2man, coords=[num, day, lat_t2m, lon_t2m],
                              dims=["num", "day", "lat", "lon"]),
        "labels": xr.DataArray(labels2, coords=[num], dims=["num"]),
        "centersnum": xr.DataArray(centersnum, coords=[classnum], dims=["classnum"]),
        "centers": xr.DataArray(centers, coords=[classnum], dims=["classnum"]),
    })

    ds_z500 = xr.Dataset({
        "z500": xr.DataArray(dayz500, coords=[num, day, lat_z500, lon_z500],
                             dims=["num", "day", "lat", "lon"]),
        "labels": xr.DataArray(labels2, coords=[num], dims=["num"]),
        "centersnum": xr.DataArray(centersnum, coords=[classnum], dims=["classnum"]),
        "centers": xr.DataArray(centers, coords=[classnum], dims=["classnum"]),
    })

    return ds_t2m, ds_z500


def process_season(season_key):
    s = config.SEASONS[season_key]
    raw_dir = config.RAW_DATA_DIR

    print(f"\n{'='*60}")
    print(f"Processing {s['name']} season")
    print(f"{'='*60}")

    member_ids = [str(i) for i in range(config.N_MEMBERS)]

    print("\n[Step 1] Combining TREFHT data...")
    combined_t2m = combine_season(raw_dir, "t2m", s["nyears"], s["fd"], s["ld"],
                                  config.SPATIAL_DOMAIN["t2m"], member_ids,
                                  season_name=s["name"])

    print("\n[Step 2] Computing T2m anomalies...")
    anomaly_ds = compute_anomaly(combined_t2m, s["nyears"], s["fd"], s["ld"],
                                 window=config.ROLLING_MEAN_WINDOW)

    print("\n[Step 3] Combining Z500 data...")
    combined_z500 = combine_season(raw_dir, "z500", s["nyears"], s["fd"], s["ld"],
                                   config.SPATIAL_DOMAIN["z500"], member_ids,
                                   season_name=s["name"])

    print("\n[Step 4] Applying land/sea mask...")
    try:
        land_mask_ds_t2m = land_sea_mask(anomaly_ds, config.LAND_SEA_MASK, mode="land")
        land_mask_ds_z500 = land_sea_mask(combined_z500, config.LAND_SEA_MASK, mode="land")
        print("  land/sea mask applied successfully")
    except Exception as e:
        print(f"  [warn] Land-sea mask failed: {e}, proceeding without mask")
        land_mask_ds_t2m = anomaly_ds
        land_mask_ds_z500 = combined_z500

    print("\n[Step 5] Detecting extreme events and clustering...")
    labels2, datalistv2, minll, n_clusters, onset_pattern = detect_extremes(
        anomaly_ds, combined_t2m, land_mask_ds_t2m,
        pct=config.PERCENTILE_THRESHOLD,
        min_consec=config.CONSECUTIVE_DAYS_FOR_EXTREME,
        min_sep=config.EVENT_SEPARATION_DAYS,
        n_clusters=config.N_CLUSTERS,
        n_normal=config.NORMAL_DAY_NUM,
    )

    print("\n[Step 6] Building lagged training datasets...")
    ds_t2m, ds_z500 = build_lagged_dataset(
        combined_t2m, combined_z500, anomaly_ds, labels2, datalistv2, minll,
        n_clusters, lead_lag=config.LEAD_LAG_DAYS,
    )

    t2m_out = os.path.join(config.LABELED_DATA_DIR, s["labeled_t2m"])
    z500_out = os.path.join(config.LABELED_DATA_DIR, s["labeled_z500"])

    ds_t2m.to_netcdf(t2m_out)
    print(f"  saved: {t2m_out}")

    ds_z500.to_netcdf(z500_out)
    print(f"  saved: {z500_out}")

    print(f"\n{'='*60}")
    print(f"Completed {s['name']}: {len(labels2)} samples, {n_clusters+1} classes")
    for cl in range(n_clusters + 1):
        count = np.sum(labels2 == cl)
        print(f"  class {cl}: {count} samples")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CESM1 data for extreme event classification")
    parser.add_argument("-s", "--season", choices=["s", "w"], default=None,
                        help="Process only summer (s) or winter (w). Default: both.")
    parser.add_argument("--raw", default=None, help="Override raw data directory")
    parser.add_argument("--out", default=None, help="Override output directory")
    args = parser.parse_args()

    if args.raw:
        config.RAW_DATA_DIR = args.raw
    if args.out:
        config.OUTPUT_DIR = args.out

    if args.season:
        process_season(args.season)
    else:
        process_season("s")
        process_season("w")