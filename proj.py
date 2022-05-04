#import sys, os

#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rls-assimilation/download'))

#import download

from msilib.schema import Error
from tkinter import N
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

px.set_mapbox_access_token(open(".mapbox_token").read())

def plot_sensor_loactions():
    df = pd.read_csv("positions_04_11_apr_updated.csv",skiprows=1,names=["lon","lat","none"])

    fig = px.scatter_mapbox(df,lat="lat",lon="lon",zoom=11)
    fig.update_layout(mapbox_style="open-street-map")


    # Show or write map commands
    #fig.show()
    #fig.write_image("figs/fig1.jpeg")
    return fig


def create_dfs():

    try:
        noise_df = pd.read_pickle("noise_df.pkl")
        pos_df = pd.read_pickle("pos_df.pkl")
        print("Noise and position data found. Want to proceed (y) or recompute the data (n)?")
        inp=input()
        if inp=="y":
            return noise_df, pos_df

    except Exception:
        pass



    pos_df = pd.read_csv("data/positions_04_11_apr_updated.csv",skiprows=1,names=["lon","lat","none"])

    t = np.arange(datetime(2022,4,4,0,0), datetime(2022,4,12,0,0), timedelta(minutes=1)).astype("datetime64[s]")
    #print(t)

    noise_df=pd.DataFrame(index=t)

    start_times=np.empty((len(pos_df),1),dtype="datetime64[s]")
    end_times=np.empty((len(pos_df),1),dtype="datetime64[s]")

    for i,idx in enumerate(pos_df.index):
        file_path=f"data/data20220404_20220411/70B3D5E39000{idx}-data.csv"
        df=pd.read_csv(file_path,skiprows=1)
        #print(idx,df.iloc[0]["Time"],df.iloc[-1]["Time"])
        df["Time"] = pd.to_datetime(df["Time"])
        start_times[i]=df.iloc[ 0]["Time"]
        end_times[i]  =df.iloc[-1]["Time"]
        df=df.groupby("Time").mean().reset_index()
        df=df.set_index("Time")
        df=df.rename({"dt_sound_level_dB":idx},axis=1)
        df=df.reindex(t)
        df=df.interpolate(method="linear",limit_area="inside")
        df=df.interpolate(method="linear",limit_area="outside",limit_direction="both")
        noise_df=pd.concat([noise_df,df],axis=1)
        
    #print(np.max(start_times))
    #print(np.min(end_times))

    #print(noise_df)

    #X=noise_df.reset_index().drop("Index").to_numpy()
    #plt.scatter(t,X)
    #plt.show()
    px.scatter(noise_df)

    noise_df.to_pickle("noise_df.pkl")
    pos_df.to_pickle("pos_df.pkl")

    return noise_df, pos_df

def split_dataframe(noise_df):
    
    day = datetime.time(datetime(2022,1,1,7,0))
    evening = datetime.time(datetime(2022,1,1,19,0))
    night = datetime.time(datetime(2022,1,1,23,0))


    day_mask = (day <= noise_df.index.time) & (noise_df.index.time < evening)
    noise_day_df = noise_df.loc[day_mask]

    evening_mask = (evening <= noise_df.index.time) & (noise_df.index.time < night)
    noise_evening_df = noise_df.loc[evening_mask]

    night_mask = np.invert(day_mask) & np.invert(evening_mask)
    noise_night_df = noise_df.loc[night_mask]

    return noise_day_df, noise_evening_df, noise_night_df

def reconstruct(noise_df):
    pass

def main():
    noise_df, pos_df = create_dfs()

    split_dataframe(noise_df)



if __name__ == "__main__":
    main()
