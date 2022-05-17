import os
from datetime import timedelta
import numpy as np
import pandas as pd

from rls_assimilation.RLSAssimilation import RLSAssimilation
from helpers import get_rmse, get_uncertainty_matrix


def prepare_data(d_path):
    all_data_df = pd.read_csv(d_path, index_col=0)
    all_data_df.index = pd.to_datetime(
        list(all_data_df.index), format="%Y-%m-%d %H:%M:%S"
    )
    all_data_df = all_data_df.sort_index()
    daily_means1 = all_data_df[f"{variable}"].resample("D").mean()
    daily_means1.index = daily_means1.index + timedelta(days=1)
    observations_source1_daily_and_hourly = pd.concat(
        [all_data_df[f"{variable}"][23:], daily_means1], axis=1
    ).ffill()
    observations_source1_daily_and_hourly.columns = [
        f"{variable}_obs_hourly",
        f"{variable}_obs_daily",
    ]

    daily_means2 = all_data_df[f"{variable}_model"].resample("D").mean()
    daily_means2.index = daily_means2.index + timedelta(days=1)
    observations_source2_daily_and_hourly = pd.concat(
        [all_data_df[f"{variable}_model"][23:], daily_means2], axis=1
    ).ffill()
    observations_source2_daily_and_hourly.columns = [
        f"{variable}_model_hourly",
        f"{variable}_model_daily",
    ]

    concatenated_sources_daily_and_hourly = pd.concat(
        [observations_source1_daily_and_hourly, observations_source2_daily_and_hourly],
        axis=1,
    )

    return concatenated_sources_daily_and_hourly


def run_test(data_path, variable, t1_in, t2_in, t_out, s1_in, s2_in, s_out):
    concatenated_sources_daily_and_hourly = prepare_data(data_path)
    n_observations = len(concatenated_sources_daily_and_hourly)

    assimilated = []
    err_assimilated = []

    # assimilation with calibration, since s1 != s2
    assimilator = RLSAssimilation(
        t_in1=t1_in, t_in2=t2_in, s_in1=s1_in, s_in2=s2_in, t_out=t_out, s_out=s_out
    )

    # assimilate
    for k in range(n_observations):
        latest_observation_source1 = concatenated_sources_daily_and_hourly[
            f"{variable}_{s1_in}_{t1_in}"
        ][k]
        latest_observation_source2 = concatenated_sources_daily_and_hourly[
            f"{variable}_{s2_in}_{t2_in}"
        ][k]
        (
            assimilated_obs_calibrated,
            err_assimilated_obs_calibrated,
        ) = assimilator.assimilate(
            latest_observation_source1,
            latest_observation_source2,
        )
        assimilated.append(assimilated_obs_calibrated)
        err_assimilated.append(err_assimilated_obs_calibrated)

    concatenated_sources_daily_and_hourly["Assimilated"] = assimilated

    # get performance metrics
    mean_unc = np.mean(err_assimilated)
    concatenated_sources_daily_and_hourly = (
        concatenated_sources_daily_and_hourly.dropna()
    )
    a_rmse = get_rmse(
        concatenated_sources_daily_and_hourly["Assimilated"].values,
        concatenated_sources_daily_and_hourly[f"{variable}_{s_out}_hourly"].values,
    )
    dh_rmse = get_rmse(
        concatenated_sources_daily_and_hourly[f"{variable}_{s_out}_daily"].values,
        concatenated_sources_daily_and_hourly[f"{variable}_{s_out}_hourly"].values,
    )

    return a_rmse / dh_rmse if dh_rmse != 0 else 1, mean_unc


def run_tests_for_variable(variable):
    data_path_dir = f"data/Europe_AQ/combined_{variable}"
    unc_ts = pd.Series(
        index=[f.replace(".csv", "") for f in os.listdir(data_path_dir)], dtype="float"
    )

    # input temporal scales
    t1 = "hourly"
    t2 = "daily"
    # output temporal scale
    t = t1

    # input spatial scales
    s1 = "obs"
    s2 = "model"
    # output spatial scale
    s = s2

    ratios = []
    mean_maus = []

    for filename in os.listdir(data_path_dir):
        ratio, mean_mau = run_test(
            f"{data_path_dir}/{filename}", variable, t1, t2, t, s1, s2, s
        )
        ratios.append(ratio)
        mean_maus.append(mean_mau)

        lat = filename.replace(".csv", "").split(";")[0]
        lon = filename.replace(".csv", "").split(";")[1]
        unc_ts[f"{lat};{lon}"] = mean_mau

    mean_ratio = round(np.mean(ratios), 3)
    sd_ratio = round(np.std(ratios), 3)
    min_ratio = round(np.min(ratios), 3)
    max_ratio = round(np.max(ratios), 3)
    print(f"Performance ratio: {mean_ratio}+-{sd_ratio} [{min_ratio};{max_ratio}]")

    # Uncomment to generate data for an uncertainty map
    # min_lat = 28.0
    # max_lat = 62.0
    # min_lon = -18.0
    # max_lon = 30.0
    # unc_map_df = get_uncertainty_matrix(min_lat, max_lat, min_lon, max_lon, 0.2, 50, unc_ts)
    # unc_map_df.to_csv(f'plots/maps/{variable}_map.csv')


variables = ["CO", "NO2", "O3", "SO2", "PM25", "PM10"]
for variable in variables:
    print(variable)
    run_tests_for_variable(variable)
