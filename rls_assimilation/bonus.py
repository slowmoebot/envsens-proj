import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rls_assimilation import RLSAssimilation
from helpers import (
    plot_data,
    print_metrics,
)

def demo_assimilation_and_plot(
    all_data_df,
    variable,
    ax_data,
    s_in1,
    s_in2,
    s_out,
    scenario_id,
):

    if s_in1 == s_in2:
        scenario = "DA2"
    elif s_out == s_in1:
        scenario = "DA3 (Model -> Station)"
    elif s_out == s_in2:
        scenario = "DA3 (Station -> Model)"
    else:
        raise ValueError("Unsupported testing parameters")

    observations_source1 = all_data_df[
        f"{variable}"
    ].values  # observations from source 1
    observations_source2 = all_data_df[
        f"{variable}_model"
    ].values  # observations from source 2
    n_observations = len(observations_source1)

    # assimilated (weighted) values and errors
    assimilated = []
    err_assimilated = []

    assimilator = RLSAssimilation(
        t_in1="hourly",
        t_in2="hourly",
        s_in1=s_in1,
        s_in2=s_in2,
        t_out="hourly",
        s_out=s_out,
    )

    # assimilate
    for k in range(n_observations):
        # Step 1: Obtain raw observations from 2 sources
        latest_observation_sensor1 = observations_source1[k]
        latest_observation_sensor2 = observations_source2[k]

        # Step 2: Assimilate
        (
            assimilated_obs_calibrated,
            err_assimilated_obs_calibrated,
        ) = assimilator.assimilate(
            latest_observation_sensor1,
            latest_observation_sensor2,
        )
        assimilated.append(assimilated_obs_calibrated)
        err_assimilated.append(err_assimilated_obs_calibrated)

    # plot and print metrics
    ax_data = plot_data(
        pd.Series(observations_source1, index=all_data_df.index),
        pd.Series(observations_source2, index=all_data_df.index),
        pd.Series(assimilated, index=all_data_df.index),
        variable,
        ax_data,
        scenario,
    )

    err1_r = None
    err2_r = None
    if scenario == "DA3 (Model -> Station)":
        err1_ar = assimilator.source1.get_all_errors()
        err2_ar = assimilator.source2.get_all_errors(force_ar_errors=True)
        err2_r = assimilator.source2.get_all_errors()
    elif scenario == "DA3 (Station -> Model)":
        err1_ar = assimilator.source1.get_all_errors(force_ar_errors=True)
        err1_r = assimilator.source1.get_all_errors()
        err2_ar = assimilator.source2.get_all_errors()
    else:
        err1_ar = assimilator.source1.get_all_errors()
        err2_ar = assimilator.source2.get_all_errors()
    err_r = err1_r if err1_r else err2_r

    print(f"{variable} metrics")
    print_metrics(
        observations_source1,
        observations_source2,
        assimilated,
        err1_ar,
        err2_ar,
        err_r,
        err_assimilated,
        scenario,
    )

    alldata=np.stack((observations_source1,observations_source2,assimilated,err1_ar,err2_ar,err_assimilated),axis=-1)
    #print(alldata)

    np.save(f"vars/alldata_{scenario_id}_{variable}.npy",alldata)

    return (ax_data,)

def test_scenario(s_in1, s_in2, s_out, scenario_id, w_0, P_0, use_err):

    np.save("temp_w0.npy",w_0)
    np.save("temp_P0.npy",P_0)
    np.save("temp_err.npy",use_err)

    data_path = "rls_assimilation/data/liivalaia_aq_meas_with_forecast.csv"
    all_data_df = pd.read_csv(data_path, index_col=0)
    all_data_df.index = pd.to_datetime(
        list(all_data_df.index), format="%Y-%m-%d %H:%M:%S"
    )
    all_data_df = all_data_df.sort_index()

    #print(all_data_df)
    variables = ["CO", "NO2", "O3", "SO2", "PM2.5", "PM10"]

    fig_data, axs_data = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))

    for idx, variable in enumerate(variables):
        (axs_data[idx % 3, idx % 2],) = demo_assimilation_and_plot(
            all_data_df,
            variable,
            axs_data[idx % 3, idx % 2],
            s_in1,
            s_in2,
            s_out,
            scenario_id,
        )
    fig_data.savefig(f"figs/data-{scenario_id}.jpg")

    #plt.show()


# NB: keep model as the second source (s_in2, not s_in1)

arr_P_0 = [0, 1, 2, 5, 10, 100]
arr_W_0 = [0, 1, 2, 5, 10, 100]
err_txt=["pred_err","est_uncert"]
w_0 = 10
P_0 = 1
use_err = 1
#for P_0 in arr_P_0:
test_scenario(s_in1="obs", s_in2="obs", s_out="obs", scenario_id=f"da2-{err_txt[use_err]}-P{P_0:03d}-w{w_0:03d}_impute",w_0=w_0,P_0=P_0,use_err=use_err)  # DA2
