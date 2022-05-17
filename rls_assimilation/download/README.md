# SILAM data acquisition

`download.py` file contains the script to download forecast data from the SILAM cloud storage.

## Installation

In the virtual environment shell, additionally install the required for downloading packages:

    pip install -r download/requirements.txt

## Examples of usage

To get the latest available forecast from a location (given by latitude and longitude in degrees) of a 
certain air quality modality, run:

    ts = get_silam_ts("CO", 59.431, 24.760, max_days=0)
    
To get all the available data (last 30 days of forecast) from a location, run:

    ts = get_silam_ts("CO", 59.431, 24.760, max_days=30)

For further information, refer to the `download.py` documentation.
