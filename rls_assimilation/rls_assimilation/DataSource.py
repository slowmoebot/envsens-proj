from typing import List, Optional
import numpy as np

from rls_assimilation.RLS import RLS


class RLSDailyAverage:
    """
    Implements RLS-based daily average upscaling of hourly estimates
    """

    def __init__(self):
        self.current_average: float = 0
        self.current_average_err: float = 0
        self.latest_daily_average: float = 0
        self.latest_daily_average_err: float = 0
        self.counter: int = 0
        self.r_model: Optional[RLS] = None

    def daily_reset(self):
        self.counter = 0
        self.current_average = 0
        self.current_average_err = 0

    def update(self, x_new_hourly: float, x_new_hourly_err: float):
        if self.counter == 24:
            if self.r_model is None:
                self.r_model = RLS()

            self.latest_daily_average = self.current_average
            self.latest_daily_average_err = self.current_average_err
            self.daily_reset()

        self.counter += 1
        prev_sum = self.current_average * (self.counter - 1)
        self.current_average = (prev_sum + x_new_hourly) / self.counter
        prev_sum_err = self.current_average_err * (self.counter - 1)
        self.current_average_err = (prev_sum_err + x_new_hourly_err) / self.counter


class DataSource:
    """
    Implements AR(1) and R(1) algorithms
    """

    def __init__(self, t_in: str, t_out: str, s_in: str, s_out: str):
        """
        :param t_in: input temporal scale (str, "hourly" or "daily")
        :param t_out: output temporal scale (str, "hourly" or "daily")
        :param s_in: input spatial scale (str)
        :param s_out: output spatial scale (str)
        """

        # resolutions
        self.t_in: str = t_in
        self.t_out: str = t_out
        self.s_in: str = s_in
        self.s_out: str = s_out

        # models
        self.ar_model: Optional[RLS] = None  # AR(1) model
        self.temporal_model: Optional[RLSDailyAverage] = (
            RLSDailyAverage() if t_in == "hourly" else None
        )
        self.spatial_r_model: Optional[RLS] = (
            RLS() if s_in != s_out else None
        )  # R(1) model

        # stored for plotting
        self.x_all = []  # raw values
        self.x_corr_all = []  # x_all with filled missing values (if there are any)
        self.x_calibrated_all = []  # R(1) model predictions
        self.ar_errors = []  # AR(1) modelling errors
        self.r_errors = []  # R(1) modelling errors
        self.dx_all = []

    def has_daily_average(self) -> bool:
        return self.temporal_model is not None

    def is_spatially_calibrated(self) -> bool:
        return self.spatial_r_model is not None

    def get_latest_data_point(self) -> float:
        return (
            self.x_calibrated_all[-1] if self.spatial_r_model else self.x_corr_all[-1]
        )

    def get_latest_error(self) -> float:
        return self.r_errors[-1] if self.spatial_r_model else self.ar_errors[-1]

    def get_all_errors(self, force_ar_errors=False) -> List[float]:
        return (
            self.r_errors
            if self.spatial_r_model and not force_ar_errors
            else self.ar_errors
        )

    def get_raw_data(self) -> List[float]:
        return self.x_all

    def get_corrected_data(self) -> List[float]:
        return (
            self.x_calibrated_all if self.is_spatially_calibrated() else self.x_corr_all
        )

    def upscale(self) -> (float, float):
        """
        Upscale the data of this source (get daily from hourly)

        Returns (upscaled data value (float), upscaled uncertainty (float))
        """

        if not self.has_daily_average():
            raise ValueError("Upscaling cannot be performed for daily data sources")

        return (
            self.temporal_model.latest_daily_average,
            self.temporal_model.latest_daily_average_err,
        )

    def downscale_other_source(
        self, x_hourly: float, other_x_daily: float, other_err_daily: float
    ) -> (float, float):
        """
        Downscale the data of the second source (get an hourly estimate from a daily one)
        using the relationship between hourly and daily of this source

        :param: x_hourly - hourly data value of this source (float)
        :param: other_x_daily - daily data value of the other source (float)
        :param: other_err_daily - daily uncertainty of the other source (float)
        Returns (other_x_hourly - downscaled data value (float), other_err_hourly - downscaled uncertainty (float))
        """
        if self.temporal_model.r_model:
            self.temporal_model.r_model.update(
                self.temporal_model.latest_daily_average, x_hourly
            )

        other_x_hourly = (
            self.temporal_model.r_model.predict(other_x_daily)
            if self.temporal_model.r_model
            else other_x_daily
        )

        sign_factor = -1 if other_err_daily < 0 else 1
        if self.temporal_model.r_model:
            other_err_hourly = float(
                np.abs(self.temporal_model.r_model.w[1]) * other_err_daily
            ) + sign_factor * np.abs(self.temporal_model.r_model.error)
        else:
            other_err_hourly = other_err_daily

        return other_x_hourly, other_err_hourly

    def impute(self, x_past: Optional[float]) -> float:
        """
        Imputes a missing data value with AR(1) prediction

        :param: x_past - the past value from the data source used as input for AR(1) model (float or None)
        Returns: imputed data value (float)
        """

        if not self.ar_model:
            if np.isnan(x_past):
                return 0
            #else:
            #    return x_past

        if not self.ar_model:
            self.ar_model = RLS()

        return self.ar_model.predict(x_past)

    def estimate(self, x_new: Optional[float]) -> (float, float):
        """
        Runs AR(1) uncertainty estimation

        :param: x_new - the latest value from the data source (float or None)
        Returns: (x_corr - imputed or raw data value (float), err - AR(1) uncertainty of x_corr (float))
        """

        self.x_all.append(x_new)  # save a raw observation
        t = len(self.x_all)  # the number of acquired data points

        # Run AR(1) estimation
        x_past = self.x_corr_all[-1] if t > 1 else np.nan
        # if np.isnan(x_new):
        #     x_corr = self.impute(x_past)  # impute (predict) if missing
        # else:
        #     x_corr = x_new
        #     if not np.isnan(x_past):
        #         if not self.ar_model:
        #             self.ar_model = RLS()  # initialise when data gets available
        #         
        #         self.ar_model.update(x_past, x_corr)


        x_corr = self.impute(x_past)
        #print(f"{x_past}, {x_corr}\n")

        if not self.ar_model:
            self.ar_model = RLS()
        
        if not np.isnan(x_new):
            self.ar_model.update(x_corr,x_new)

        # obtain error of the AR(1) model
        err = self.ar_model.error if self.ar_model else 0
        dx = self.ar_model.error if self.ar_model else 0

        self.x_corr_all.append(x_corr)  # save a corrected (raw or imputed) data value
        self.ar_errors.append(err)  # save the latest error
        self.dx_all.append(dx)

        return x_corr, err

    def calibrate(self, x_corr: float, err: float, x_ref: float):
        """
        Run spatial R(1) calibration

        :param: x_corr - a value being calibrated (float)
        :param: err - uncertainty of the value being calibrated (float)
        :param: x_ref - reference value for calibration (float)

        Returns (x_calibrated - calibrated data value (float), r_err - uncertainty of x_calibrated (float))
        """
        if len(self.x_calibrated_all) < 1:
            self.x_calibrated_all.append(x_corr)
            self.r_errors.append(err)
            return x_corr, err

        # Step 1: Predict
        x_calibrated = self.spatial_r_model.predict(x_corr)
        self.x_calibrated_all.append(x_calibrated)
        sign_factor = -1 if err < 0 else 1
        r_err = float(np.abs(self.spatial_r_model.w[1]) * err) + sign_factor * np.abs(
            self.spatial_r_model.error
        )
        self.r_errors.append(r_err)

        # Step 2: Update
        self.spatial_r_model.update(x_corr, x_ref)

        return x_calibrated, r_err
