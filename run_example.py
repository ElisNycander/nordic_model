"""
Example how to run the model for 1 year. For an example how to run several years at once, see case_study_example()
in case_studies.py.

"""

from nordic_model import Model
import matplotlib.pyplot as plt

if __name__ == "__main__":

    db_path = 'D:/Data/' # path to .db files
    data_path = './Data' # path to excel files (/Data)
    res_path = 'D:/ModelRun' # results stored here

    year = 2020
    casename = f'example_run_{year}'

    m = Model(name=casename, path=res_path, db_path=db_path,data_path=data_path)
    m.default_options()

    m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']

    m.opt_start = f'{year}0101'
    m.opt_end = f'{year}1231'
    m.opt_costfit_tag = f'{year}'
    m.opt_capacity_year = year
    m.opt_weather_year = year

    m.opt_solver = 'gurobi'
    m.opt_api = 'gurobi'

    m.fopt_plots = {
         'gentype': True,
         'gentot': True,
         'gentot_bar': False,
         'renewables': False,
         'transfer_internal': True,
         'transfer_external': True,
         'reservoir': False,
         'price': False,
         'losses': False,
         'load_curtailment': False,
         'inertia': False,
         'hydro_duration': False,
         'wind_curtailment': False}
    # run single year
    m.run(save_model=True)

