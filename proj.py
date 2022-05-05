import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

px.set_mapbox_access_token(open(".mapbox_token").read())


def get_sensor_locations():
    df = pd.read_csv("positions_04_11_apr_updated.csv", skiprows=1, names=["lon", "lat", "none"])
    # fig = px.scatter_mapbox(df,lat="lat",lon="lon",zoom=11)
    # fig.update_layout(mapbox_style="open-street-map")

    # Show or write map commands
    # fig.show()
    # fig.write_image("figs/fig1.jpeg")

    return df[["lon", "lat"]]


def create_dfs():
    try:
        noise_df = pd.read_pickle("vars/noise_df.pkl")
        pos_df = pd.read_pickle("vars/pos_df.pkl")
        print("Noise and position data found. Want to proceed (y) or recompute the data (n)?")
        inp = input()
        if inp == "y":
            return noise_df, pos_df

    except Exception:
        pass

    pos_df = get_sensor_locations()

    t = np.arange(datetime(2022, 4, 4, 0, 0), datetime(2022, 4, 12, 0, 0), timedelta(minutes=1)).astype("datetime64[s]")
    # print(t)

    noise_df = pd.DataFrame(index=t)

    start_times = np.empty((len(pos_df), 1), dtype="datetime64[s]")
    end_times = np.empty((len(pos_df), 1), dtype="datetime64[s]")

    for i, idx in enumerate(pos_df.index):
        file_path = f"data/data20220404_20220411/70B3D5E39000{idx}-data.csv"
        df = pd.read_csv(file_path, skiprows=1)
        # print(idx,df.iloc[0]["Time"],df.iloc[-1]["Time"])
        df["Time"] = pd.to_datetime(df["Time"])
        start_times[i] = df.iloc[0]["Time"]
        end_times[i] = df.iloc[-1]["Time"]
        df = df.groupby("Time").mean().reset_index()
        df = df.set_index("Time")
        df = df.rename({"dt_sound_level_dB": idx}, axis=1)
        df = df.reindex(t)
        # other interpolation methods can be applied by exchanging linear
        df = df.interpolate(method="linear", limit_area="inside")
        df = df.interpolate(method="linear", limit_area="outside", limit_direction="both")
        noise_df = pd.concat([noise_df, df], axis=1)

    # print(np.max(start_times))
    # print(np.min(end_times))

    # print(noise_df)

    # X=noise_df.reset_index().drop("Index").to_numpy()
    # plt.scatter(t,X)
    # plt.show()
    px.scatter(noise_df)

    noise_df.to_pickle("vars/noise_df.pkl")
    pos_df.to_pickle("vars/pos_df.pkl")

    return noise_df, pos_df


def split_dataframe(noise_df):
    day = datetime.time(datetime(2022, 1, 1, 7, 0))
    evening = datetime.time(datetime(2022, 1, 1, 19, 0))
    night = datetime.time(datetime(2022, 1, 1, 23, 0))

    day_mask = (day <= noise_df.index.time) & (noise_df.index.time < evening)
    noise_day_df = noise_df.loc[day_mask]

    evening_mask = (evening <= noise_df.index.time) & (noise_df.index.time < night)
    noise_evening_df = noise_df.loc[evening_mask]

    night_mask = np.invert(day_mask) & np.invert(evening_mask)
    noise_night_df = noise_df.loc[night_mask]

    return noise_day_df, noise_evening_df, noise_night_df


def reconstruct(noise_df, n):
    u, s, vh = np.linalg.svd(noise_df.to_numpy())

    print(n, u.shape, s.shape, vh.shape)

    mat = np.dot(u[:, :n] * s[:n], vh[:n, :])

    res_df = pd.DataFrame(mat, index=noise_df.index, columns=noise_df.columns)

    return res_df


def save_rmse(noise_df, name, step):
    n_samples = len(noise_df.columns) * len(noise_df.index)
    rmse = np.empty(len(noise_df.columns), dtype="float")

    for i, n in enumerate(np.arange(1, len(noise_df.columns), step)):
        noise_df_re = reconstruct(noise_df, n)

        errs = np.power(noise_df_re.to_numpy() - noise_df.to_numpy(), 2)
        rmse[i] = np.sum(errs) / n_samples
        print(name, np.sum(errs) / n_samples)

    np.save(f"vars/{name}.npy", rmse)


def read_regions():
    file_dict = {1: "data/region 1.csv",
                 2: "data/region 2.csv",
                 3: "data/region 3.csv"}

    reg_dict = {}

    for key, val in file_dict.items():
        reg_dict[key] = pd.read_csv(val, header=0).index.to_list()

    return reg_dict


def get_station_data():
    noise_df, res_df = create_dfs()
    noise_day_df, noise_evening_df, noise_night_df = split_dataframe(noise_df)
    reg_dict = read_regions()

    res_df["full_mean"] = noise_df.mean()
    res_df["full_std"] = noise_df.std()
    res_df["day_mean"] = noise_day_df.mean()
    res_df["day_std"] = noise_day_df.std()
    res_df["evening_mean"] = noise_evening_df.mean()
    res_df["evening_std"] = noise_evening_df.std()
    res_df["night_mean"] = noise_night_df.mean()
    res_df["night_std"] = noise_night_df.std()

    res_df.loc[reg_dict[1], "reg"] = 1
    res_df.loc[reg_dict[2], "reg"] = 2
    res_df.loc[reg_dict[3], "reg"] = 3

    return res_df


def main():
    res_df = get_station_data()
    fig = px.scatter_mapbox(res_df,
                            lat="lat",
                            lon="lon",
                            zoom=11,
                            mapbox_style="open-street-map",
                            color="full_mean"
                            )
    fig.show()

    # print(noise_df[reg_dict[1]])

    # rmse_full = np.load("vars/full_every.npy")

    # for e in rmse_full:
    #    print(e)

    # print(rmse_full)
    # plt.plot(rmse_full)
    # plt.show()
    # save_rmse(noise_df,"full_every",1)
    # save_rmse(noise_day_df,"day_every",1)
    # save_rmse(noise_evening_df,"evening_every",1)
    # save_rmse(noise_night_df,"night_every",1)


if __name__ == "__main__":
    main()
