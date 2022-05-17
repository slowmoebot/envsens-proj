from math import sin, cos, sqrt, atan2, radians
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 22})


def get_rmse(arr1, arr2):
    return np.sqrt(((arr1 - arr2) ** 2).mean()).round(2)


def get_uncertainty_stats(err):
    return f"{np.mean(err).round(2)} ± {np.std(err).round(2)} [{np.min(err).round(2)}; {np.max(err).round(2)}]"


def print_metrics(
    s1,
    s2,
    assimilated,
    err1_ar,
    err2_ar,
    err_r,
    err_assimilated,
    scenario,
):
    s1[np.isnan(s1)] = 0
    s2[np.isnan(s2)] = 0

    # Root Mean Squared Errors
    #print(f"RMSE (Station and Model): {get_rmse(s1, s2)}")
    print(f"RMSE (Station and {scenario}): {get_rmse(s1, assimilated)}")
    #print(f"RMSE (Model and {scenario}): {get_rmse(s2, assimilated)}")

    # Mean Absolute Uncertainties
    #print(f"MAU (Station): {np.mean(np.abs(err1_ar)).round(2)}")
    #print(f"MAU (Model): {np.mean(np.abs(err2_ar)).round(2)}")

    if scenario == "DA3 (Model -> Station)":
        print(f"MAU (Model calibrated): {np.mean(np.abs(err_r)).round(2)}")
    elif scenario == "DA3 (Station -> Model)":
        print(f"MAU (Station calibrated): {np.mean(np.abs(err_r)).round(2)}")

    print(f"MAU ({scenario}): {np.mean(np.abs(err_assimilated)).round(2)}")


def plot_data(
    s1,
    s2,
    assimilated,
    variable,
    ax_data,
    scenario,
):
    ax_data.set_title(f"{variable}")
    ax_data.plot(
        s1.index,
        s1.values,
        color="red",
        marker="X",
        linewidth=0,
        markersize=6,
    )
    ax_data.plot(
        s2.index,
        s2.values,
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_data.plot(
        assimilated.index,
        assimilated.values,
        color="black",
        linestyle="-",
        linewidth=2,
    )
    ax_data.set_xlabel("Date")
    ax_data.set_ylabel("ug/m³")
    ax_data.grid()

    every_nth = 2
    for n, label in enumerate(ax_data.xaxis.get_ticklabels()):
        if n % every_nth != 1:
            label.set_visible(False)

    if variable == "CO":
        ax_data.legend(
            [
                "Station",
                "Model",
                scenario,
            ]
        )

    return ax_data


def plot_data2(
    s1_hourly,
    s1_daily,
    s2_hourly,
    s2_daily,
    assimilated,
    variable,
    ax_data,
):
    ax_data.set_title(f"{variable}")
    ax_data.plot(
        s1_hourly.index,
        s1_hourly.values,
        color="red",
        marker="X",
        linewidth=0,
        markersize=10,
    )
    ax_data.plot(
        s1_daily.index,
        s1_daily.values,
        color="blue",
        linestyle="-",
        linewidth=3,
    )
    ax_data.plot(
        s2_hourly.index,
        s2_hourly.values,
        color="gray",
        linestyle="-",
        linewidth=3,
    )
    ax_data.plot(
        s2_daily.index,
        s2_daily.values,
        color="black",
        linestyle="-",
        linewidth=3,
    )
    assimilated.plot(
        s2_daily.index,
        s2_daily.values,
        color="green",
        linestyle="-",
        linewidth=3,
    )

    ax_data.set_xlabel("Date")
    ax_data.set_ylabel("ug/m³")
    ax_data.grid()

    if variable == "CO":
        ax_data.legend(
            [
                "Station hourly",
                "Station daily",
                "Model hourly",
                "Model daily",
                "DA",
            ]
        )

    return ax_data


def get_distance_between_points(lat1, lon1, lat2, lon2):
    r = 6373.0  # Earth radius in km

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return r * c


def get_distances_from_stations(lat, lon, stations):
    dist_dict = dict()
    for coord in stations:
        st_lat = float(coord.split(";")[0])
        st_lon = float(coord.split(";")[1])
        dist_dict[coord] = get_distance_between_points(lat, lon, st_lat, st_lon)

    return dist_dict


def get_stations_within_distance(lat, lon, unc_ts, dist_km):
    dist_dict = get_distances_from_stations(lat, lon, unc_ts.index.values)
    unc_interp = 0

    for key in dist_dict.keys():
        d = dist_dict[key]
        if dist_dict[key] <= dist_km:
            w = dist_dict[key] / dist_km
            mau = unc_ts[key]
            unc_interp += w * mau

    return unc_interp if unc_interp != 0 else None


def get_uncertainty_matrix(min_lat, max_lat, min_lon, max_lon, step, dist_km, unc_ts):
    unc_map_df = pd.DataFrame(dtype=float)
    lats = np.arange(min_lat, max_lat, step)
    lons = np.arange(min_lon, max_lon, step)
    for i in range(len(lats)):
        for j in range(len(lons)):
            lat = lats[i]
            lon = lons[j]

            unc = get_stations_within_distance(lat, lon, unc_ts, dist_km)

            unc_map_df = unc_map_df.append(
                {"lat": lat, "lon": lon, "MAU": unc}, ignore_index=True
            )

    return unc_map_df
