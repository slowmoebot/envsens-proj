from typing import Optional
import numpy as np

from rls_assimilation.DataSource import DataSource


class RLSAssimilation:
    """
    Least-squares assimilation of data from 2 data sources

    :param t_in1: temporal scale of source1 (str, "hourly" or "daily")
    :param t_in2: temporal scale of source2 (str, "hourly" or "daily")
    :param s_in1: spatial scale of source1 (str)
    :param s_in2: spatial scale of source1  (str)
    :param t_out: temporal scale of assimilation output (str, "hourly" or "daily")
    :param s_out: spatial scale of assimilation output (str)
    """

    def _validate(
        self, t_in1: str, t_in2: str, s_in1: str, s_in2: str, t_out: str, s_out: str
    ):
        # 1) Supported temporal scales: 'hourly' and 'daily
        err_t = (
            t_in1
            if t_in1 not in ["hourly", "daily"]
            else (
                t_in2
                if t_in2 not in ["hourly", "daily"]
                else (t_out if t_out not in ["hourly", "daily"] else None)
            )
        )
        if err_t is not None:
            raise NotImplementedError(
                f'Temporal scale {err_t} is not supported. Supported temporal scales are "hourly" and "daily".'
            )

        # 2) Output spatial scale must be equal to at least one of the input spatial scales
        if not (s_out == s_in1 or s_out == s_in2):
            raise ValueError(
                f"Output spatial scale {s_out} must be equal to at least one of the input spatial scales {s_in1} and/or {s_in2}"
            )

        # 3) Output temporal scale must be equal to at least one of the input temporal scales
        if not (t_out == t_in1 or t_out == t_in2):
            raise ValueError(
                f"Output temporal scale {t_out} must be equal to at least one of the input temporal scales {t_in1} and/or {t_in2}"
            )

    def __init__(
        self, t_in1: str, t_in2: str, s_in1: str, s_in2: str, t_out: str, s_out: str
    ):
        # Validate prerequisites
        self._validate(t_in1, t_in2, s_in1, s_in2, t_out, s_out)
        # Create objects for 2 data sources
        self.source1: DataSource = DataSource(t_in1, t_out, s_in1, s_out)
        self.source2: DataSource = DataSource(t_in2, t_out, s_in2, s_out)

    def _align_scales_of_sources(
        self,
        _source1_obs: float,
        _err_source1: float,
        _source2_obs: float,
        _err_source2: float,
    ) -> (float, float, float, float):
        # Obtain data values and uncertainties in the t_out and s_out scales
        source1_obs = _source1_obs
        err_source1 = _err_source1
        source2_obs = _source2_obs
        err_source2 = _err_source2

        # Spatial calibration
        if (
            self.source1.is_spatially_calibrated()
            and not self.source2.is_spatially_calibrated()
        ):
            source1_obs, err_source1 = self.source1.calibrate(
                source1_obs, err_source1, source2_obs
            )
        elif (
            self.source2.is_spatially_calibrated()
            and not self.source1.is_spatially_calibrated()
        ):
            source2_obs, err_source2 = self.source2.calibrate(
                source2_obs, err_source2, source1_obs
            )

        # Update daily averages for hourly data sources
        if self.source1.has_daily_average():
            self.source1.temporal_model.update(source1_obs, err_source1)
        if self.source2.has_daily_average():
            self.source2.temporal_model.update(source2_obs, err_source2)

        # Temporal scaling
        if self.source1.t_in != self.source1.t_out:
            if self.source1.t_in == "hourly":
                source1_obs, err_source1 = self.source1.upscale()
            else:
                source1_obs, err_source1 = self.source2.downscale_other_source(
                    source2_obs, source1_obs, err_source1
                )
        elif self.source2.t_in != self.source2.t_out:
            if self.source2.t_in == "hourly":
                source2_obs, err_source2 = self.source2.upscale()
            else:
                source2_obs, err_source2 = self.source1.downscale_other_source(
                    source1_obs, source2_obs, err_source2
                )

        return source1_obs, err_source1, source2_obs, err_source2

    def assimilate(
        self, obs1: Optional[float], obs2: Optional[float]
    ) -> (float, float):
        """
        Assimilate values for 2 data sources with unknown uncertainty

        :param: obs1 - value from the first data source (float or None)
        :param: obs2 - value from the second data source (float or None)

        Returns (assimilated_obs - assimilated value (float), err_assimilated_obs - uncertainty of assimilated_obs (float))
        """

        # Step 1: Pre-process observations and estimate AR(1) errors
        source1_obs, err_source1 = self.source1.estimate(obs1)
        source2_obs, err_source2 = self.source2.estimate(obs2)

        #print(self.source1.dx_all)
        #print(self.source2.dx_all)

        # Step 2: Temporal and spatial calibration
        (
            source1_obs,
            err_source1,
            source2_obs,
            err_source2,
        ) = self._align_scales_of_sources(
            source1_obs, err_source1, source2_obs, err_source2
        )

        # Step 3: Assimilation
        try:
            k = (err_source2**2) / (err_source1**2 + err_source2**2)

        except ZeroDivisionError:
            k = 1

        assimilated_obs = k * source1_obs + (1 - k) * source2_obs

        err_assimilated_obs = np.sqrt(
            (k * err_source1) ** 2 + ((1 - k) * err_source2) ** 2
        )

        return assimilated_obs, err_assimilated_obs
