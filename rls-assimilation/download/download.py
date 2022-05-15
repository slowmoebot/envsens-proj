import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import boto3
from botocore.handlers import disable_signing


def get_forecast_from_silam_zarr(date_str, modality, day, version="v5_7_1"):
    """
    Obtain forecast of specified parameter from SILAM for the whole world in zarr format

    :param date_str: date of forecast generation: 8 digits (YYYYMMDD), e.g. 21210101 (string)
    :param modality: CO, NO2, NO, O3, PM10, PM25, SO2, airdens (string)
    :param day: one of 0, 1, 2, 3, 4 (number)
    :param version: "v5_7_1" by default, if needed, check version on
    http://fmi-opendata-silam-surface-zarr.s3-website-eu-west-1.amazonaws.com/?prefix=global/

    :return: dataset of forecasts (xarray dataset)
    """

    bucket_name = f"fmi-opendata-silam-surface-zarr"
    key = f"global/{date_str}/silam_glob_{version}_{date_str}_{modality}_d{day}.zarr"

    tmp_dir = "/tmp"
    tmp_file = tmp_dir + "/" + key

    if not os.path.exists(os.path.dirname(tmp_file)):
        os.makedirs(os.path.dirname(tmp_file))

    def download(bucket_name, key, dst_root="/tmp"):
        """Download zarr directory from S3"""

        resource = boto3.resource("s3")
        resource.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)

        bucket = resource.Bucket(bucket_name)
        for object in bucket.objects.filter(Prefix=key):
            dst = dst_root + "/" + object.key
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            resource.meta.client.download_file(bucket_name, object.key, dst)

    # download data
    download(bucket_name, key)

    # read dataset from the downloaded file
    ds = xr.open_zarr(tmp_file, consolidated=False)

    return ds


def get_series_from_location(ds, modality, approx_lat, approx_lon):
    """
    Obtain time series from the whole world dataset from a specified location

    :param ds: whole world dataset (xarray dataset)
    :param modality: CO, NO2, NO, O3, PM10, PM25, SO2, airdens (string)
    :param approx_lat: location of interest - latitude in degrees (float)
    :param approx_lon: location of interest - longitude in degrees (float)

    :return: localised time series (pandas time series)
    """

    def find_closest_to(arr, val):
        return arr.flat[np.abs(arr - val).argmin()]

    # find the closest model cell coordinates and obtain data from that location
    lat = find_closest_to(ds[modality].lat.values, approx_lat)
    lon = find_closest_to(ds[modality].lon.values, approx_lon)

    times = [val.values for val in list(ds[modality].time)]
    data = ds[modality].sel(lat=lat, lon=lon).values

    return pd.Series(index=times, data=data)


def get_all_days_series(start_date, modality, lat, lon):
    """
    Obtain 5-day forecast of [modality] from [start_date] from location [lat; lon]

    :param start_date: date of forecast generation (datetime)
    :param modality: CO, NO2, NO, O3, PM10, PM25, SO2, airdens (string)
    :param lat: location of interest - latitude in degrees (float)
    :param lon: location of interest - longitude in degrees (float)

    :return: 5 concatenated time series of forecasts (pandas series)
    """

    def get_date_str(start_date):
        month_str = f'{start_date.month if len(str(start_date.month)) == 2 else f"0{start_date.month}"}'
        day_str = f'{start_date.day if len(str(start_date.day)) == 2 else f"0{start_date.day}"}'
        return f"{start_date.year}{month_str}{day_str}"

    # transform date into 8 digits (YYYYMMDD) string
    date_str = get_date_str(start_date)

    # obtain forecasts for each of 5 days and concatenate them
    series_list = []
    for d in range(5):
        ds = get_forecast_from_silam_zarr(date_str, modality, day=d)
        ts = get_series_from_location(ds, modality, lat, lon)
        series_list.append(ts)

    return pd.concat(series_list, axis=0)


def get_silam_ts(modality, lat, lon, max_days=30):
    """
    Obtain time series of [modality] generated by SILAM during the last [max_days] days from location [lat; lon]

    :param modality: CO, NO2, NO, O3, PM10, PM25, SO2, airdens (string)
    :param lat: location of interest - latitude in degrees (float)
    :param lon: location of interest - longitude in degrees (float)
    :param max_days: number of days (get all data from 0 - today, 30 - 30 days ago (number)
    NB: 30 days is maximum stored on SILAM cloud

    :return: time series of forecasts (pandas series)
    """

    all_series = []
    for offset_days in range(0, max_days + 1):
        start_date = datetime.datetime.now() - datetime.timedelta(offset_days)
        series = get_all_days_series(start_date, modality, lat, lon)
        all_series.append(series)

    # build a dataframe from 5-day forecasts
    df = pd.DataFrame(columns=[0, 1, 2, 3, 4])
    for ts in all_series:
        for idx, val in ts.items():
            if idx in list(df.index):
                for col in df.columns:
                    if np.isnan(df.loc[idx, col]):
                        df.loc[idx, col] = val
                        break
            else:
                df.loc[idx, 0] = val

    df = df.sort_index()

    # take series of the latest available estimates
    for idx, row in df.iterrows():
        for col in df.columns:
            if col != "0":
                if not np.isnan(row[col]):
                    df.loc[idx, "0"] = row[col]
                else:
                    break

    silam_ts = df["0"]
    format = "%Y-%m-%d %H:%M:%S"
    silam_ts.index = pd.to_datetime(list(silam_ts.index), format=format)

    return silam_ts
