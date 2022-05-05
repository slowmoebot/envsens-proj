import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, distance_matrix
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator

px.set_mapbox_access_token(open(".mapbox_token").read())

mapbox_access_token = open(".mapbox_token").read()

def get_sensor_locations():
    df = pd.read_csv("positions_04_11_apr_updated.csv", skiprows=1, names=["lon", "lat", "none"])
    # fig = px.scatter_mapbox(df,lat="lat",lon="lon",zoom=11)
    # fig.update_layout(mapbox_style="open-street-map")

    # Show or write map commands
    # fig.show()
    # fig.write_image("figs/fig1.jpeg")

    return df[["lon", "lat"]]

def read_dfs():
    noise_df = pd.read_pickle("vars/noise_df.pkl")
    pos_df = pd.read_pickle("vars/pos_df.pkl")
    return noise_df, pos_df

def create_dfs():
    pos_df = get_sensor_locations()

    t = np.arange(datetime(2022, 4, 4, 0, 0), datetime(2022, 4, 12, 0, 0), timedelta(minutes=1)).astype("datetime64[s]")
    # print(t)

    noise_df = pd.DataFrame(index=t)


    for idx in pos_df.index:
        file_path = f"data/data20220404_20220411/70B3D5E39000{idx}-data.csv"
        df = pd.read_csv(file_path, skiprows=1)
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.groupby("Time").mean().reset_index()
        df = df.set_index("Time")
        df = df.rename({"dt_sound_level_dB": idx}, axis=1)
        df = df.reindex(t)
        # other interpolation methods can be applied by exchanging linear
        df = df.interpolate(method="linear", limit_area="inside")
        df = df.interpolate(method="linear", limit_area="outside", limit_direction="both")
        noise_df = pd.concat([noise_df, df], axis=1)

    noise_df.to_pickle("vars/noise_df.pkl")
    pos_df.to_pickle("vars/pos_df.pkl")


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
        rmse[i] = np.sqrt(np.sum(errs) / n_samples)
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


def get_station_data(noise_df):
    a, res_df = read_dfs()
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

def simple_idw(x, y, z, xi, yi):

    x = np.asarray(x)

    print(x,x.shape)

    dist = distance_matrix()

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi


def idw(x,y,z,x_test,y_test):

    n_x,n_y = x_test.shape


    X = np.concatenate((np.reshape(x_test,(-1,1)),np.reshape(y_test,(-1,1))),axis=1)
    Y = np.c_[x,y]

    invD = np.power(distance_matrix(X,Y),-1)

    spl = np.divide(invD @ z, invD @ np.ones_like(z))

    return np.reshape(spl,(n_x,n_y))


def rbf(x,y,z,x_test,y_test,eps = 1):

    n_x,n_y = x_test.shape

    X = np.concatenate((np.reshape(x_test,(-1,1)),np.reshape(y_test,(-1,1))),axis=1)
    Y = np.c_[x,y]

    D_int = distance_matrix(Y,Y)

    phi_int = np.exp(-eps * np.power(D_int,2))

    weights = np.linalg.solve(phi_int,z)

    print(weights)

    D = distance_matrix(X,Y)
    phi = np.exp(-eps * np.power(D,2))

    #spl = phi @ weights

    interp = LinearNDInterpolator((x,y),z)

    spl = interp(np.reshape(x_test,(-1,1)),np.reshape(y_test,(-1,1)))


    return np.reshape(spl,(n_x,n_y))


def spacial_interpolation(res_df,res_var="full_mean"):
    
    n_edge = 101
    n_sensors = len(res_df)

    lon_min=res_df["lon"].min()
    lon_max=res_df["lon"].max()
    lat_min=res_df["lat"].min()
    lat_max=res_df["lat"].max()

    d_lat=lat_max-lat_min
    d_lon=lon_max-lon_min

    lons = np.linspace(lon_min-0.05*d_lon,lon_max+0.05*d_lon,n_edge)
    lats = np.linspace(lat_min-0.05*d_lat,lat_max+0.05*d_lat,n_edge)

    Lon, Lat = np.meshgrid(lons,lats)

    Spl_idw = idw(res_df["lon"].to_numpy(), res_df["lat"].to_numpy(), res_df[res_var],Lon,Lat)
    Spl_rbf = rbf(res_df["lon"].to_numpy(), res_df["lat"].to_numpy(), res_df[res_var],Lon,Lat)

    fig = go.Figure()
    
    """
    fig.add_densitymapbox(
        lon=res_df["lon"],
        lat=res_df["lat"],
        z=res_df[res_var],
        #mapbox_style="open-street-map",
        #zoom = 11
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            accesstoken=mapbox_access_token,
            #zoom=10
        ),
    )
    """

    
    fig.add_contour(
        x=lons,
        y=lats,
        z=Spl_rbf,
        opacity=1,
        contours=dict(
            start=res_df[res_var].min(),
            end=res_df[res_var].max(),
            size=2,
        )
    )
    
    fig.add_scatter(
        x=res_df["lon"],
        y=res_df["lat"],
        mode="markers"
    )
    
    fig.show()


    #tri = Delaunay(res_df[["lon","lat"]].to_numpy())

    #print(tri)

def main():

    """
    res_df = get_station_data()
    fig = px.scatter_mapbox(res_df,
                            lat="lat",
                            lon="lon",
                            zoom=11,
                            mapbox_style="open-street-map",
                            color="full_mean"
                            )
    fig.show()
    fig.write_image("figs/fig1.jpeg")
    """

    noise_df, pos_df = read_dfs()

    res_df = get_station_data(noise_df)

    spacial_interpolation(res_df)



if __name__ == "__main__":
    main()
