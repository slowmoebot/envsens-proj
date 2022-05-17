# RLS-based data assimilation with unknown uncertainty

Implementation of the Recursive Least Squares (RLS)-based uncertainty quantification and least-squares data assimilation
algorithm described in paper [Lightweight Assimilation of Open Urban Ambient Air Quality Monitoring Data and Numerical Simulations 
with Unknown Uncertainty](https://www.researchgate.net/publication/359872882_Lightweight_Assimilation_of_Open_Urban_Ambient_Air_Quality_Monitoring_Data_and_Numerical_Simulations_with_Unknown_Uncertainty).

Python: 3.*

## Testing

To be able to reproduce the results presented in the paper, install requirements from `requirements.txt`.
It will install the packages needed to read and plot the data.

Project package dependencies can be installed using `pip` and a virtual environment manager:

Create and activate a virtual environment:

    python -m venv rls-venv
    source rls-venv/bin/activate

In the virtual environment, install requirements from the `requirements.txt` file:

    pip install -r requirements.txt
   
After installing the requirements, run the script `example1.py`:
 
    python example1.py

The script will generate plots in `plots/Liivalaia/` and print metrics for the Tallinn dataset 
`data/liivalaia_aq_meas_with_forecast.csv`.
    
## Repository content

`rls_assimilatiion` is the lightweight package for uncertainty quantification and least-squares data 
assimilation.

The used data is stored in the `data/` directory, plots are generated to `plots/` directory.

Directory `download/` contains script to download data from the SILAM cloud storage.
