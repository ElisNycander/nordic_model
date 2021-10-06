# -*- coding: utf-8 -*-
""" Simple multi-area model for Nordic electricity market

Created on Wed Jan 16 11:31:07 2019

@author: elisn


Notes:
1 - For conversion between dates (YYYYMMDD:HH) and weeks (YYYY:WW) weeks are counted as starting during the first hour
in a year and lasting 7 days, except for the last week which covers the remaining hours in the year. Thus all years
are assumed to have 52 weeks. This definition is not according to ISO calendar standard but is legacy from the
first version of the model, probably changing it would not significantly change the results. Also note that the
MAF inflow data used also does not follow ISO calendar standard for weeks but counts weeks as starting with Sundays.
2 - It is not known if the ENTSO-E reservoir data corresponds to the reservoir level at the beginning/end of the week.
This can decrease the accuracy of the model for short time periods but does not affect much when simulating a whole year
A request has been made to find the answer from ENTSO-E
3 - For the exchange GB-NL, February 20-27 2016, the flows and scheduled exchanges are outside the implicitly
allocated day ahead capacity, it's not known why

"""


### EXTERNAL LIBRARIES ###
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import datetime
import os
import csv
from itertools import islice
import pyomo.environ as pye
from contextlib import redirect_stdout

##############################################################
######### MODULES FROM POWER_DATABASES #######################
##############################################################
import maf_hydro_data
import maf_pecd_data
import entsoe_transparency_db as entsoe
from help_functions import find_overlimit_events, compact_xaxis_ticks, \
    week_to_range, str_to_date, find_str, intersection, duration_curve, interp_time, \
    interpolate_weekly_values, time_to_bin, err_func, curtailment_statistics

### INTERNAL MODULES ###
from pyomo_model import PyomoModel
from gurobi_model import GurobiModel
from pyomo_init_model import PyomoInitModel
from offer_curves import SupplyCurve
from model_definitions import MWtoGW, GWtoMW, cm_per_inch, std_fig_size, area_to_country, country_to_areas, entsoe_type_map, synchronous_areas, colors, \
    nordpool_capacities, generators_def, solver_executables, solver_stats, bidz2maf_pecd, co2_price_ets, \
    new_trans_cap, GWtoTW, TWtoGW, all_areas
from help_classes import EmptyObject, Error
from week_conversion import WeekDef


class Model:
    """ Contains all data processing not related to specific solver api (gurobi/pyomo)

    NAMING CONVENTIONS: 
    df_xxx - dataframe obtained from external database

    TIME CONVENTIONS:
    For average energy quantities, time stamp marks beginning of the (hourly) interval. This is consistent with
    convention in databases, since the beginning of the hour has been used to time stamp hourly data.

    starttime - beginning of first period in model
    endtime - end of last period in model
    timerange - all hours modelled (beginning of hour)
    idx_time - index of timerange, used to create time set in optimization model
    timerange_p1 - all hours including endtime hour

    Note: the data used is retrieved for all hours in timerange plus one extra hour, to allow for interpolation of
    the data to higher resolution
    """

    def __init__(self,name='default',path='D:/NordicModel/Results',db_path='D:/Data',
                 data_path='D:/NordicModel/InputData'):

        self.name = name

        self.data_path = Path(data_path)
        self.db_path = Path(db_path)
        self.res_path = Path(path) / name
        self.fig_path = self.res_path / 'Figures'
        self.root_path = self.res_path # points to root directory of this model

        self.res_path.mkdir(exist_ok=True,parents=True)
        self.fig_path.mkdir(exist_ok=True,parents=True)

        self.runs = [] # store results from multiple model runs
        self.res_time = {} # store runtime info

    def update_path(self,path='D:/NordicModel/Results/case'):
        """ Update path where figures and results are stored, without changing root path """
        self.res_path = Path(path)
        self.fig_path = self.res_path / 'Figures'
        self.res_path.mkdir(exist_ok=True)
        self.fig_path.mkdir(exist_ok=True)

    def default_options(self):
        """ Set default options for model """
        
        ############# BASIC OPTIONS ##############
        self.opt_solver = 'ipopt' # solver to use, must be installed
        self.opt_api = 'pyomo' # pyomo/gurobi (gurobi api only works if solver is also gurobi)
        self.opt_solver_opts = {} # options to pass to solver (with pyomo api)
        self.opt_start = '20180101'
        self.opt_end = '20180108'
        self.opt_weather_year = 2016 # used to get maf data, inflow data, and solar merra data
        self.opt_load_scale = 1 # scale load by this factor
        self.opt_loss = 0 # Fraction of energy lost in transmission
        self.opt_nonnegative_data = ['inflow']
        self.opt_countries = ['SE','DK','NO','FI','EE','LT','LV','PL','DE','NL','GB'] # modelled countries
        self.opt_use_maf_pecd = False # use solar and wind data from MAF2020
        self.opt_impute_limit = 30 # maximum number of values to interpolate in data
        self.opt_impute_constant = { # constants used to impute remaining missing values in input data
            'exchange':0, # for external exchanges
            'solar':0,
        }
        self.opt_run_initialization = False # run low resolution model to get values for initialization
        self.opt_init_delta = 168
        # Note: initialization is useful for some solvers (e.g. ipopt) but may not be for others (e.g. gurobi)

        self.opt_db_files = {
            'capacity':'capacity.db',
            'prices':'prices.db',
            'exchange':'exchange.db',
            'gen':'gen.db',
            'unit':'unit.db',
            'load':'load.db',
            'reservoir':'reservoir.db',
            'inflow':'inflow.db',
            'maf_hydro':'maf_hydro.db',
            'maf_pecd':'maf_pecd.db',
        }
        self.opt_err_labl = 'MAE' # should be consistent with the error computed in err_func

        ########## COST OPTIONS ##########################
        self.opt_costfit_tag = '2019' # use this costfit from the input parameters
        self.opt_hydro_cost = False # include fitted hydro costs, not properly implemented
        self.opt_default_thermal_cost = 40 # default value for thermal cost
        self.opt_loadshed_cost = 3000 # cost for demand curtailment
        self.opt_nuclear_cost = 7.35 # default value for nuclear cost
        self.opt_wind_cost = 1 # low wind cost in EUR/MWh to favour wind curtailment over solar
        self.opt_use_var_cost = True # use variable costs
        # Source for variable cost data: data['costfit_shifted']['tag']
        # replace extreme cost fits (e.g. decreasing mc or very sharply increasing mc with fuel-based constant MC)
        self.opt_overwrite_bad_costfits = True
        self.opt_c2_min = 1e-5
        self.opt_c2_max = 0.5

        # specify co2 price, this is added to the price coefficient MC(p)=k*p+m+(co2_price-co2_price(offset_year))
        self.opt_co2_price = None
        self.opt_co2_price_offset_year = 2016 # if set to year, this assumes m already contains the cost for that year

        ############ TECHNICAL LIMITS #########################
        self.opt_capacity_year = 2019 # use generation capacity from entsoe for this year
        self.opt_hvdc_max_ramp = 600 # 600 MW/hour
        self.opt_pmax_type = 'capacity'
        self.opt_pmax_type_hydro = 'stats'
        # Options for pmax: 'stats' - from gen_stats.xlsx (production statistics)
        #                   'capacity' - from entsoe capacity per type database
        # For hydro the source for the maximum capacity is chosen separately
        self.opt_pmin_zero = False # put pmin = 0

        ######### NUCLEAR OPTIONS ################
        self.opt_nucl_min_lvl = 0.65 # nuclear can ramp down to this level
        self.opt_nucl_ramp = None # overwrite nuclear ramp rate (%/hour)
        self.opt_nucl_add_cap = {
            'SE3':0,
            'FI':0,
            'DE':0,
        } # add this firm capacity to nuclear generation

        # option to compute nuclear max levels from individual units for some areas, can be used to deactivate certain
        # nuclear reactors in order to simulate scenarios, requires production data for individual units
        self.opt_nucl_individual_units = []
        # exclude these nuclear reactors when deciding maximum generation levels - only possible with opt_nucl_individual_units
        self.opt_nucl_units_exclude = []
        #self.opt_nucl_units_exclude = ['Ringhals block 1 G11','Ringhals block 1 G12','Ringhals block 2 G21','Ringhals block 2 G22']

        ######### HYDRO OPTIONS #################
        self.opt_reservoir_offset = 168
        self.opt_reservoir_data_normalized = True # use normalized reservoir data
        self.opt_default_inflow = 100
        self.opt_default_inflow_area = { # GWh/week, per area
            'DE':346, # 180 TWh yearly production
            'PL':45,
            'GB':107,
        }
        self.opt_use_maf_inflow = False # use MAF inflow data or inflow calculated from ENTSO-E data
        # inflow interpolation:
        # constant (i.e. constant for one week)
        # linear (linear ramp rate between weeks)
        self.opt_inflow_interp = 'linear'
        self.opt_hydro_daily = False # daily reservoir constraints (instead of hourly)
        self.opt_reservoir_start_fill = 0.5 # if reservoir data does not exist, assume default filling value
        self.opt_reservoir_end_fill = 0.5
        # share of inflow which is run of river, if no data available
        self.opt_ror_fraction = {
            'SE1':0.13,
            'SE2':0.21,
            'SE3':0.27,
            'SE4':0.3,
            'NO1':0.25,
            'NO2':0,
            'NO3':0,
            'NO4':0,
            'NO5':0,
            'FI':0.27,
            'LV':0.4,
            'LT':0.5,
            'PL':0.8,
            'DE':0.9,
            'GB':0.4,
        }
        self.opt_reservoir_capacity = { # GWh
            'NO1':6507,
            'NO2':33388,
            'NO3':8737,
            'NO4':19321,
            'NO5':16459,
            'SE1':13688,
            'SE2':15037,
            'SE3':2517,
            'SE4':216,
            'FI':4512,
            'LT':12.2,
            'LV':11.2,
            'PL':2.3,
            'DE':1263,
            'GB':26.4,
        }
        # pumping capacity
        self.opt_pump_capacity = { # in MW, from MAF data
            'PL':1660,
            'DE':7960,
            'GB':2680,
            'NO1':130,
            'NO2':430,
            'NO3':70,
            'NO5':470,
            'LT':720,
        }
        self.opt_pump_reservoir = { # in GWh
            'PL':6.3,
            'DE':20,
        }
        # pumping efficiency
        self.opt_pump_efficiency = 0.75

        ############# RESERVE OPTIONS ################
        self.opt_use_reserves = False # include reserve requirements
        self.opt_country_reserves = False # reserves by country instead of by area (more flexibility)
        self.opt_reserves_fcrn = { # this is the allocation of 600 MW FCR-N
            'SE':245,
            'NO':215,
            'DK':0,
            'FI':140,
        }
        self.opt_reserves_fcrd = 1200 # FCR-D, allocated in same proportion as FCR-N

        ######## EXTERNAL AREAS OPTIONS #################
        # the price will be set for these price areas, and the export/import will be variable instead of fixed
        self.opt_set_external_price = ['DE','PL']
        self.opt_default_prices = {
            'PL':40, # use this price for external connections if no other is avaialable
            'RU':40,
            'DE':40,
            'NL':40,
            'GB':40,
        }
        self.opt_exchange_data_type = 'flow'

        ########### TRANSFER CAPACITY OPTIONS #####################
        self.opt_use_var_exchange_cap = True
        self.opt_nominal_capacity_connections = [('NL','GB'),]
        # these connections will always use nomianl exchange capacity
        self.opt_min_exchange_cap = 100 # minimum variable transfer capacity (MW)
        # may be set to >= 2018 to include additional future transmission capacity,
        # from new_trans_cap in model_definitions
        self.opt_exchange_cap_year = None

        ########## WIND OPTIONS #############
        self.opt_wind_scale_factor = {
            'SE1':1,
            'SE2':1,
            'SE3':1,
            'SE4':1,
        }
    
        self.opt_wind_capacity_onsh = {
            'DK1':3725,
            'DK2':756,
            'EE':329,
            'FI':2422,
            'LT':540,
            'LV':84,
            'NO1':166,
            'NO2':1145,
            'NO3':1090,
            'NO4':668,
            'NO5':0,
            'SE1':1838,
            'SE2':3849,
            'SE3':2780,
            'SE4':1581,
            'PL':5952,
            'NL':3973,
            'DE':53338,
            'GB':14282,
        }
        self.opt_wind_capacity_offsh = {
            'DK1':1277,
            'DK2':423,
            'EE':0,
            'FI':0,
            'LT':0,
            'LV':0,
            'NO1':0,
            'NO2':0,
            'NO3':0,
            'NO4':0,
            'NO5':0,
            'SE1':0,
            'SE2':0,
            'SE3':0,
            'SE4':0,
            'PL':0,
            'NL':1709,
            'DE':7504,
            'GB':10383,
        }

        ########### SOLAR OPTIONS #############
        # Note: the solar capacities only apply if opt_use_maf_pecd is True, otherwise ENTSO-E production data is used for solar
        # manually specify solar capacity for areas:
        self.opt_solar_cap_by_area = {
            'DK1':878, # from ENTSO-E
            'DK2':422,
            'EE':164,
            'FI':215,
            'LT':169,
            'LV':11,
            'SE1':9, # from Energiåret 2020 (energiföretagen)
            'SE2':67,
            'SE3':774,
            'SE4':240,
            'PL':1310,
            'NL':5710,
            'DE':48376,
            'GB':13563,
        }
        # if solar capacity for an area is not specified, the aggregated value
        # for that country is used, weighted by the areas share of total load
        self.opt_solar_cap_by_country = { # from IRENA Capacity Statistics 2020
            'DK':1079,
            'FI':215,
            'NO':90,
            'SE':644,
            'LV':3,
            'LT':103,
            'EE':107,}


        ########## INERTIA OPTIONS ####################
        self.opt_use_inertia_constr = False     # inertia constraints
        self.opt_min_kinetic_energy = 113 # GWs
        # calculation of kinetic energy: Ek = H*P/(cf*pf)
        # inertia constants from Persson (2017) Kinetic Energy Estimation in the Nordic System
        self.opt_inertia_constants = {
            'SE':{'Hydro':4.5,'Thermal':2.9,'Nuclear':6.2,},
            'NO':{'Hydro':2.9,'Thermal':2.5,},
            'FI':{'Hydro':2.8,'Thermal':4.4,'Nuclear':6.6,},
            'DK':{'Thermal':4.5,},
        }
        # assumption about power factor pf
        self.opt_inertia_pf = {
            'SE':{'Hydro':0.9,'Thermal':0.9,'Nuclear':0.9,},
            'NO':{'Hydro':0.9,'Thermal':0.9,},
            'FI':{'Hydro':0.9,'Thermal':0.9,'Nuclear':0.9,},
            'DK':{'Thermal':0.9,},
        }
        # assumption about capacity factor cf
        self.opt_inertia_cf = {
            'SE':{'Hydro':0.8,'Thermal':1,'Nuclear':1,},
            'NO':{'Hydro':0.8,'Thermal':1,},
            'FI':{'Hydro':0.8,'Thermal':1,'Nuclear':1,},
            'DK':{'Thermal':1,},
        }

        ####### ROUNDING VALUES ##############
        self.opt_bound_cut = { # round values below this threshold to zero, to avoid small coefficients
            'max_SOLAR':1e-4,
            'max_WIND':1e-4,
            'min_PG':1e-4,
        }

        ######## FIGURE OPTIONS ##################
        self.fopt_no_plots = False
        self.fopt_plots = {
            'gentype':True,
            'gentot':True,
            'gentot_bar':False,
            'renewables':False,
            'transfer_internal':True,
            'transfer_external':True,
            'reservoir':False,
            'price':False,
            'losses':False,
            'load_curtailment':False,
            'inertia':False,
            'hydro_duration':False,
            'wind_curtailment':False,
        }
        self.fopt_plot_weeks = []
        self.fopt_use_titles = True
        self.fopt_show_rmse = True # also show absolute RMSE on fopt_plots
        self.fopt_eps = False
        self.fopt_print_text = False # print model to text file
        self.fopt_print_dual_text = False # print dual to text file
        self.fopt_dpi_qual = 1000
        # control inset in plot
        self.fopt_inset_date = None
        self.fopt_inset_days = 5


        self.fopt_calc_rmse = { # some rmse calculations need additional data
            'price':True,
            'transfer':True
        }
        self.fopt_rmse_transfer_data_type = 'flow'

        ##### OPTIONS TO PRINT OUTPUT ######
        self.opt_print = {
            'init':True,
            'solver':True,
            'setup':True,
            'postprocess':True,
            'check':True,
        }

        self.default_pp_opt()

    def default_pp_opt(self):
        ########## OPTIONS CONTROLLING POST PROCESSING ###############
        self.pp_opt = EmptyObject()

        self.pp_opt.get_vars = ['SPILLAGE','PG','RES','X1','X2','WIND','XEXT','LS','SOLAR','HROR','PUMP','REL','PRES']
        self.pp_opt.inst_vars = ['RES','PRES']
        self.pp_opt.daily_vars = ['RES','SPILLAGE'] # daily variables if opt_hydro_daily is True
        # Note: duals only obtained only if the constraint exists (some constraints are optional)
        self.pp_opt.get_duals = ['POWER_BALANCE','RESERVOIR_BALANCE','HVDC_RAMP','GEN_RAMP',
                                 'RESERVES_UP','RESERVES_DW','FIX_RESERVOIR','INERTIA']
        self.pp_opt.get_cur_vars = ['WIND','SOLAR','HROR']

    def effective_reservoir_range(self):
        # effective ranges, based on min and max reservoir values from entso-e data
        self.opt_reservoir_capacity = { # GWh
            'SE1':11326,
            'SE2':13533,
            'SE3':1790,
            'SE4':180,
            'FI':2952,
            'NO1':6078,
            'NO2':21671,
            'NO3':7719,
            'NO4':14676,
            'NO5':14090,
            'LT':11.8,
            'LV':9.4,
            'DE':2430,
            'PL':2800,
            'GB':4100,
        }

    def vre_cap_2016(self):
        """ Set wind and solar capacities to values from 2016, for validation of model with MAF data for this year """
        pass
        # SOLAR CAPACITY
        self.opt_solar_cap_by_area = {
            'DK1':421, # from ENTSO-E
            'DK2':180,
            'PL':77,
            'NL':1429,
            'DE':40679,
            'GB':11914,
        }

        # if solar capacity for an area is not specified, the aggregated value
        # for that country is used, weighted by the areas share of total load
        self.opt_solar_cap_by_country = { # from IRENA Capacity Statistics
            'DK':851,
            'FI':39,
            'NO':27,
            'SE':153,
            'LV':1,
            'LT':70,
            'EE':10,
        }

        # MAF WIND CAPACITY
        self.opt_wind_capacity_onsh = {
            'DK1':2966,
            'DK2':608,
            'EE':375,
            'FI':2422,
            'LT':366,
            'LV':55,
            'NO1':0,
            'NO2':261,
            'NO3':361,
            'NO4':251,
            'NO5':0,
            'SE1':524,
            'SE2':2289,
            'SE3':2098,
            'SE4':1609,
            'PL':5494,
            'NL':3284,
            'DE':45435,
            'GB':10833,
        }

        self.opt_wind_capacity_offsh = {
            'DK1':843,
            'DK2':428,
            'EE':0,
            'FI':0,
            'LT':0,
            'LV':0,
            'NO1':0,
            'NO2':0,
            'NO3':0,
            'NO4':0,
            'NO5':0,
            'SE1':0,
            'SE2':0,
            'SE3':0,
            'SE4':0,
            'PL':0,
            'NL':357,
            'DE':4000,
            'GB':5293,
        }

    def run(self,save_model=False):
        """ Run single case of model, for current settings """
        pass
        self.res_time = {}
        t_0 = time.time()
        self.setup()
        self.res_time['pre'] = time.time() - t_0
        t__0 = time.time()
        self.setup_child_model()
        self.res_time['cm'] = time.time() - t__0
        self.solve()
        t__0 = time.time()
        self.post_process()
        self.res_time['post'] = time.time() - t__0
        self.res_time['tot'] = time.time() - t_0

        if save_model:
            self.save_model()

    def run_years(self,years=range(2015,2017),append=False,save_full_model=False):
        """ Run model using weather data for multiple years between start and end
        save_full_model: Save full model using save_model for start year in root path
        """
        start = years[0]
        self.opt_weather_year = start
        self.update_path(self.root_path/f'{start}')
        # run first instance of model
        self.run()
        self.save_model_run(append=append)

        if save_full_model:
            self.update_path(self.root_path)
            self.save_model()

        # update weather data and run remaining instances
        for year in years[1:]:
            self.update_path(self.root_path/f'{year}')
            self.re_run_year(year=year)
            self.save_model_run(append=append)

    def re_run_year(self,year=2015):
        """ Update the weather year and re-run model """
        print(f'---- RE-RUN YEAR {year} -----')
        self.res_time = {}
        t_0 = time.time()
        self.opt_weather_year = year
        self.setup_weather_indices()
        self.get_inflow_data()
        self.setup_inflow()
        self.setup_run_of_river()
        self.setup_inflow_feasibility()
        self.max_HROR = {
            (a,t):self.ror_hourly.at[self.timerange[t],a]*MWtoGW for a in self.ror_areas for t in self.idx_time
        }
        self.setup_solar()
        self.setup_wind()
        self.max_SOLAR = {
            (a,t):self.solar.at[self.timerange[t],a] for a in self.solar_areas for t in self.idx_time
        }
        self.max_WIND = {
            (a,t):self.wind.at[self.timerange[t],a]*self.opt_wind_scale_factor[a]
            for a in self.wind_areas for t in self.idx_time
        }
        for name in ['max_WIND','max_SOLAR']:
            self.round_bound(name)

        #%%
        if self.opt_run_initialization:
            self.run_init_model()
        #%%
        self.res_time['pre'] = time.time() - t_0
        t_1 = time.time()
        self.cm.update_inflow()
        self.cm.update_ror(self.max_HROR)
        self.cm.update_solar(self.max_SOLAR)
        self.cm.update_wind(self.max_WIND)

        self.res_time['cm'] = time.time() - t_1
        #%% rerun model
        self.solve()
        t_1 = time.time()
        self.post_process()
        self.res_time['post'] = time.time() - t_1
        self.res_time['tot'] = time.time() - t_0

        print(f'------ FINISHED YEAR {year} --------')

    def load_results_years(self,vars=['res_PG','res_LS'],years=None):
        """ Get given results for all yearly runs"""

        res = {
            v:{} for v in vars
        }

        exist_years = []
        for y in [y for y in os.listdir(self.root_path) if os.path.isdir(self.root_path / y)]:
            try:
                exist_years.append(int(y))
            except Exception:
                pass
        if years is None:
            years = exist_years
        else:
            years = [y for y in exist_years if y in years]

        # get results from all runs
        for y in years:
            self.load_model_run(y)
            for v in vars:
                res[v][y] = self.__getattribute__(v)
        return res

    def round_bound(self,name):
        prt = self.opt_print['setup']
        if name in self.opt_bound_cut:
            thrs = self.opt_bound_cut[name]
            dic = self.__getattribute__(name)
            count = 0
            for i,val in dic.items():
                if val > 0 and val < thrs:
                    dic[i] = 0
                    count += 1
            if count and prt:
                print(f'Rounded {count} values to zero in {name}')

    def save_model(self):
        """
        Dump all model results to pickle file. Also save options, gen data etc., as well as self.runs
        Can produce very large file if several runs are stored in self.runs
        The valus saved are sufficient to rerun all plot functions, after first calling setup_data
        """
        d = {}
        save_vars = ['runs','ror_areas','generators_def','hydrores','areas','wind_areas','solar_areas','pump_res_areas',
                     'pump_areas','ror_reserve_areas','nuclear_areas','resareas','syncareas','gen_in_area',
                     'xtrans_int','xtrans_ext','rescountries','reservoir_capacity','pump_reservoir','fixed_transfer_connections',
                     'fixed_price_connections','area_sep_str','solar_capacity',
                     ]
        vars = [v for v in dir(self) if v.split('_',1)[0] in ['res','gen','idx','opt','fopt','dual','max','min'] or v in save_vars]
        for v in vars:
            d[v] = self.__getattribute__(v)
        with open(self.root_path/f'results.pkl','wb') as f:
            pickle.dump(d,f)

    def save_model_run(self,append=False):
        """
        Dump results from current model run in results.pkl
        If append=True, results are also appended to list in self.runs
        Storing many runs in self.runs can consume lots of memory, so it may
        be better just to save the pickle files and load them when needed
        """
        # save_entities = ['inflow_hourly','weeks','inflow','inflow_hourly_tmp','ror_hourly']
        save_entities = []
        run = {
            v:self.__getattribute__(v) for v in [ v for v in dir(self) if v.split('_',1)[0] == 'res' or v in save_entities]
        }
        run['opt_weather_year'] = self.opt_weather_year
        if append:
            self.runs.append(run)
        with open(self.res_path/f'results.pkl','wb') as f:
            pickle.dump(run,f)

    def load_model(self):
        with open(self.res_path/f'results.pkl','rb') as f:
            d = pickle.load(f)
        for v in d:
            self.__setattr__(v,d[v])

    def load_model_run(self,year=2015):
        self.res_path = self.root_path / f'{year}'
        self.load_model()
        self.res_path = self.root_path

    def redo_plots(self):
        print('----- REDO PLOTS -----')
        self.load_model()
        self.setup_indices()
        self.setup_weather_indices()
        self.setup_data()
        self.get_rmse_data()
        self.plot_figures()
        
    def setup_child_model(self):
        """ Create the Pyomo/Gorubi model object """
        api = self.opt_api
        solver = self.opt_solver
        # Choose child model "cm" class depending on api type
        if api == 'gurobi' and solver == 'gurobi':
            self.cm = GurobiModel(name=self.name)
        else:
            if api == 'gurobi':
                print(f'WARNING: Can only use gurobi api with gurobi, using pyomo api!')
            self.cm = PyomoModel()

        self.cm.setup_opt_problem(self)

    def setup(self):
        pass
        prt = self.opt_print['setup']
        self.vars_df_up_bound = {
            'WIND':['wind_areas','idx_time'],
            'SOLAR':['solar_areas','idx_time'],
            'LS':['areas','idx_time'],
            'HROR':['ror_areas','idx_time'],
        }
        print('----- SETUP -------------')

        self.setup_indices()
        self.setup_weather_indices()

        self.setup_transmission()
        if prt:
            print('----- SETUP DATA --------')
        self.setup_data()
        if prt:
            print('----- SETUP GEN ---------')
        self.setup_gen()
        if prt:
            print('----- SETUP RESERVES ----')
        self.setup_reserves()
        if prt:
            print('----- SETUP HYDRO -------')
        self.setup_hydro()
        if prt:
            print('----- SETUP WIND --------')
        self.setup_wind()
        if prt:
            print('----- SETUP SOLAR -------')
        self.setup_solar()
        if prt:
            print('----- SETUP RESERVOIR ---')
        self.setup_reservoir_values()
        if prt:
            print('----- SETUP INFLOW ------')
        self.setup_inflow()
        if prt:
            print('----- SETUP ROR  --------')
        self.setup_run_of_river()
        self.setup_inflow_feasibility()
        if prt:
            print('----- SETUP BOUNDS  -----')
        self.setup_bounds()

        if self.opt_run_initialization:
            self.run_init_model()

        print('----- SETUP COMPLETE ----')

        self.print_hydro_table()
        self.print_renewable_table()

    def solve(self):
        """ Solve model """
        print(' ----- STARTING SOLVER -----')
        prt = self.opt_print['solver']
        solver = self.opt_solver

        if not hasattr(self,'cm'):
            print('Model does not have child model, run "setup_child_model"')
            return None
        elif type(self.cm) is PyomoModel:

            ## DECLARE DUAL
            if not hasattr(self.cm,'dual'):
                self.cm.dual = pye.Suffix(direction=pye.Suffix.IMPORT)
            ## SOLVE MODEL
            if solver in solver_executables: # give explicit solver path
                opt = pye.SolverFactory(solver,executable=solver_executables[solver],options=self.opt_solver_opts)
            else:
                opt = pye.SolverFactory(solver,options=self.opt_solver_opts)
            res = opt.solve(self.cm, tee=prt)
            if 'Time' in res['solver'][0]:
                self.res_time['solver'] = res['solver'][0]['Time']
            else:
                self.res_time['solver'] = np.nan
            self.res_stats = {
                name:res['problem'][0][solver_stats['pyomo'][name]] for name in solver_stats['pyomo']
            }
        else:
            if not prt:
                self.cm.gm.setParam('OutputFlag',0)
            self.cm.gm.optimize()
            self.res_time['solver'] = self.cm.gm.Runtime
            self.res_stats = {
                name:self.cm.gm.getAttr(solver_stats['gurobi'][name]) for name in solver_stats['gurobi']
            }
        print(' ----- FINISHED SOLVER -----')


    def post_process(self):
        """ Post-processing of optimization results and plotting of figures """
        print('----- POST PROCESS  ------')
        prt = self.opt_print['postprocess']
        ############### RESULTS ##########################
        self.res_residuals = {} # residuals to check supply == demand

        self.res_rmse_area = pd.DataFrame(dtype=float,index=self.areas,columns=['Prod','Hydro','Thermal','Nuclear','Price'])
        
        self.res_rmse_intcon = pd.DataFrame(index=self.xtrans_int.index,columns=['From','To','RMSE'])
        # self.res_rmse_intcon.loc[:,['From','To']] = self.xtrans_int.loc[:,['from','to']]
        self.res_rmse_intcon['From'] = self.xtrans_int['from']
        self.res_rmse_intcon['To'] = self.xtrans_int['to']
        self.res_rmse_extcon = pd.DataFrame(index=self.xtrans_ext.index,columns=['From','To','RMSE'])
        # self.res_rmse_extcon.loc[:,['From','To']] = self.xtrans_ext.loc[:,['from','to']]
        self.res_rmse_extcon['From'] = self.xtrans_ext['from']
        self.res_rmse_extcon['To'] = self.xtrans_ext['to']
        
        self.res_rmse_area_norm = self.res_rmse_area.copy()
        self.res_rmse_intcon_norm = pd.Series(index=self.xtrans_int.index)
        self.res_rmse_extcon_norm = pd.Series(index=self.xtrans_ext.index)

        # if given path, override
        self.get_df_bounds()
        self.get_results_from_child()
        self.get_rmse_data()
        if prt:
            print('----- POST CALC.  -------')
        self.post_process_calculations()
        # some more curtailment stats
        print('----- PLOT FIGURES  -----')
        self.plot_figures()
        #self.plot_offer_curves(self.supply_curve_hour)
        self.print_rmse()

        # writing output takes too long for large models
        if self.fopt_print_dual_text:
            with open(self.res_path /  'dual.txt','w') as f:
                with redirect_stdout(f):
                    self.dual.display()
        if self.fopt_print_text:
            with open(self.res_path /  'model.txt','w') as f:
                with redirect_stdout(f):
                    self.pprint()

        if self.opt_print['check']:
            print('----- CHECK RESULTS  ----')
            print(f'Maximum residual: {max([self.res_residuals[area] for area in self.res_residuals])}')
            print(f'Average losses: {np.mean(self.res_losses):0.4f} %')
            print('Errors:')
            print(f'Production: {self.res_rmse_area["Prod"].mean():0.4f}')
            print(f'Hydro: {self.res_rmse_area["Hydro"].mean():0.4f}')
            print(f'Thermal: {self.res_rmse_area["Thermal"].mean():0.4f}')
            print(f'Transfer: {self.res_rmse_intcon["RMSE"].mean():0.4f}')
            print(f'External transfer: {self.res_rmse_extcon["RMSE"].mean():0.4f}')
            print(f'Price: {self.res_rmse_area["Price"].mean():0.4f}')



    def run_init_model(self):
        t_0 = time.time()
        prt = self.opt_print['init']
        if prt:
            print('------- RUN INIT MODEL --------')
        self.setup_init_model()
        self.solve_init_model()
        self.postprocess_init_model()
        if prt:
            print('------- INIT MODEL COMPLETE ---')
        self.res_time['ini'] = time.time() - t_0
    def setup_init_model(self):
        """
        This sets up a low resolution model, which is solved to get values with which to initialize the hourly model
        :return:
        """
        print_output = self.opt_print['init']
        self.ini = EmptyObject()
        t0 = time.time()
        delta = self.opt_init_delta
        if print_output:
            print(f'Time step of {delta} hours')
        pass
        self.timerange_lr = [self.timerange[t] for t in range(0,self.nPeriods,delta)]
        # delta = 168 # resolution of model

        # compute number of periods
        self.nPeriods_lr = int(np.ceil(self.nPeriods / delta))
        # map hour indices to periods
        p2i = {}
        i2p = {}
        for pidx in range(self.nPeriods_lr):
            # h2p[i] = range()
            p2i[pidx] = range(pidx*delta,min((pidx+1)*delta,self.nPeriods))
        for pidx in p2i:
            for i in p2i[pidx]:
                i2p[i] = pidx
        self.p2i = p2i
        self.i2p = i2p


        entities = [e for e in ['solar','wind','exchange','exchange_capacity','demand','ror_hourly','inflow_hourly'] \
                    if hasattr(self,e)]
        for name in entities:
            df = self.__getattribute__(name)
            self.__setattr__(f'{name}_lr',df.resample(f'{delta}H').mean())
        if self.opt_use_var_cost:
            self.gen_c1_lr = self.gen_c1.resample(f'{delta}H').mean()
            self.gen_c2_lr = self.gen_c2.resample(f'{delta}H').mean()

        # interpolate reservoir values for initialization
        self.reservoir_interp_lr = interp_time(self.timerange_lr,self.reservoir_fix)

        self.setup_bounds(lowres=True)
        self.cmi = PyomoInitModel()
        self.cmi.setup_opt_problem(self)

        self.ini.time_setup = time.time() - t0

    def solve_init_model(self):
        print_output = self.opt_print['init']
        solver = self.opt_solver
        self.cmi.dual = pye.Suffix(direction=pye.Suffix.IMPORT)
        ## SOLVE MODEL
        if solver in solver_executables:
            opt = pye.SolverFactory(solver,executable=solver_executables[solver])
        else:
            opt = pye.SolverFactory(solver)
        t0 = time.time()
        self.ini.res = opt.solve(self.cmi, tee=print_output)
        self.ini.time_solve = time.time() - t0

    def postprocess_init_model(self):
        pass
        print_output = self.opt_print['init']
        t0 = time.time()

        """ Get result variables, duals, and bounds from optimization problem """
        mo = self.ini
        mo.obj = self.cmi.get_objective_value()

        # read results into dataframes
        if print_output:
            print('Reading results into Panda data frames')
        for v in self.pp_opt.get_vars:
            entity = self.cmi.get_variable(f'var_{v}')
            # convert to date index
            entity.index = self.timerange_lr
            mo.__setattr__(f'res_{v}',entity)

        # increment time index of instantaneous variables
        for var in [v for v in self.pp_opt.inst_vars if v in self.pp_opt.get_vars]:
            entity = mo.__getattribute__(f'res_{var}')
            entity.index += datetime.timedelta(hours=self.opt_init_delta)

        # get dual variables
        if print_output:
            print('Getting dual variables')
        for v in self.pp_opt.get_duals:
            constr = f'constr_{v}'
            if hasattr(self.cmi,constr):
                entity = self.cmi.get_dual(constr)
                # convert to date index
                if v not in ['FIX_RESERVOIR']:
                    entity.index = self.timerange_lr
                mo.__setattr__(f'dual_{constr}',entity)
                # dic[f'dual_{constr}'] = entity

        # interpolate reservoir values
        mo.reservoir_interp = pd.DataFrame(dtype=float,index=self.timerange_p1,columns=self.hydrores)
        mo.reservoir_interp.loc[self.timerange[0],:] = self.reservoir_fix.loc[self.timerange[0],:]
        mo.reservoir_interp.loc[self.timerange_p1[-1],:] = self.reservoir_fix.loc[self.timerange_p1[-1],:]
        mo.reservoir_interp.loc[mo.res_RES.index[:-1],:] = np.array(mo.res_RES.loc[mo.res_RES.index[:-1],:])
        mo.reservoir_interp.interpolate(inplace=True)

        mo.time_post = time.time() - t0

    def setup_indices(self):
        prt = self.opt_print['setup']
        self.starttime = self.opt_start + ':00'
        self.endtime = (str_to_date(self.opt_end) + datetime.timedelta(hours=24)).strftime('%Y%m%d') + ':00'

        # defined quantities from options
        self.areas = []
        for c in self.opt_countries:
            for a in country_to_areas[c]:
                self.areas.append(a)
        self.single_area_countries = [c for c in self.opt_countries if country_to_areas[c].__len__() == 1]
        self.multi_area_countries = [c for c in self.opt_countries if country_to_areas[c].__len__() > 1]
        self.syncareas = [a for a in self.areas if a in synchronous_areas]

        self.country_to_areas = { # country to areas for countries included in model
            c:country_to_areas[c] for c in self.opt_countries
        }
        self.area_to_country = {
            a:area_to_country[a] for a in self.areas
        }

        self.hydrores = [area for area in self.areas if 'Hydro' in generators_def[area]]


        # note: period with index 0 is starting period
        # self.start = str_to_date(self.starttime) + datetime.timedelta(hours=-1)
        # self.end = str_to_date(self.endtime)
        self.timerange = pd.date_range(start=str_to_date(self.starttime),
                                       end=str_to_date(self.endtime)+datetime.timedelta(hours=-1),freq='H')
        self.timerange_p1 = pd.date_range(start=str_to_date(self.starttime),
                                            end=str_to_date(self.endtime),freq='H')

        self.nPeriods = self.timerange.__len__()
        self.idx_time = range(self.nPeriods)

        # day_fmt = '%Y%m%d'
        self.daysrange_p1 = pd.date_range(start=self.timerange_p1[0],end=self.timerange_p1[-1],freq='D')
        self.daysrange = self.daysrange_p1[:-1]
        self.nDays = self.daysrange_p1.__len__() - 1
        self.idx_day = range(self.nDays)

        # map hours to days
        self.hour2day = {
            t:int(np.floor_divide(t,24)) for t in self.idx_time
        }
        self.day2hour = {
            d:[t for t in self.idx_time if self.hour2day[t] == d] for d in self.idx_day
        }



        #%% set start/end time for weather data (wind,solar,hydro)
        start_year = int(self.opt_start[:4])
        self.start_year = start_year



    def setup_weather_indices(self):
        """ Setup indices related to weather year, which effects the inflow data and (for the MAF data and Merra solar data)
        the wind and solar production
        """

        start_year = self.start_year
        # check that weather year is within maf range
        if (self.opt_use_maf_inflow or self.opt_use_maf_pecd) and \
                (self.opt_weather_year > 2016 or self.opt_weather_year < 1982):
            print(f'WARNING: No maf data for {self.opt_weather_year}, setting weather year to 2016')
            self.opt_weather_year = 2016

        self.weather_year_diff = self.opt_weather_year - start_year
        sfmt = '%Y%m%d:%H'
        self.starttime2 = (datetime.datetime(year=start_year+self.weather_year_diff,
                                             month=int(self.starttime[4:6]),
                                             day=int(self.starttime[6:8]))).strftime(sfmt)
        self.endtime2 = (datetime.datetime(year=int(self.endtime[:4])+self.weather_year_diff,
                                           month=int(self.endtime[4:6]),
                                           day=int(self.endtime[6:8]))).strftime(sfmt)

        # week to date conversion
        if self.opt_use_maf_inflow:
            # maf data starts on Sunday
            self.wd = WeekDef(week_start=7,proper_week=False)
        else:
            # use ISO week definition
            self.wd = WeekDef(week_start=4,proper_week=True)

        # get week range covering whole simulation period
        # Note: since the simulated year may be leap year, we may need one more day of data, hence get extra week
        start = self.starttime2
        end = (datetime.datetime.strptime(self.endtime2,sfmt)+datetime.timedelta(days=7)).strftime(sfmt)
        self.weeks = self.wd.range2weeks(start,end,sout=True)
        self.widxs = self.wd.range2weeks(start,end,sout=False)
        # find index offset, we will interpolate inflow data for the whole range in weeks/widxs
        # and then use df[inflow_offset:inflow_offset+nPeriods] as the inflow data
        self.inflow_offset = int((datetime.datetime.strptime(self.starttime2,sfmt)-self.widxs[0]).seconds/3600)

        dstart = str_to_date(self.starttime2)
        self.daysrange_weather_year = pd.date_range(start=dstart,end=dstart+datetime.timedelta(days=self.nDays-1),freq='D')

    def get_inflow_data(self):

        if self.opt_use_maf_inflow: # get MAF inflow data
            # get both weekly and daily maf inflow data
            self.inflow,self.inflow_daily_maf = self.maf_hydro_db.select_inflow_bidz_wrap(starttime=self.weeks[0],
                                                                   endtime=self.weeks[-1],
                                                                   areas=self.hydrores,
                                                                   wd=self.wd,date_index=True)
        else: # entsoe inflow
            self.inflow = self.inflow_db.select_inflow_data(starttime=self.weeks[0],
                                                               endtime=self.weeks[-1],
                                                               areas=self.hydrores,
                                                        table='inflow',wd=self.wd,date_index=True)

    def db_exists(self,db='prices.db'):
        # Check that database exists
        pass
        if not os.path.isfile(self.data_path / db):
            raise Error(f"Database {db} does not exist!")

    def setup_data(self):

        prt = self.opt_print['setup']

        # Check that databases exist
        for db in [f for f in self.opt_db_files if f != 'unit']:
            self.db_exists(self.db_path / self.opt_db_files[db])

        self.price_db = entsoe.Database(db=self.db_path / self.opt_db_files['prices'])
        self.exchange_db = entsoe.Database(db=self.db_path / self.opt_db_files['exchange'])
        self.load_db = entsoe.Database(db=self.db_path / self.opt_db_files['load'])
        self.reservoir_db = entsoe.Database(db=self.db_path / self.opt_db_files['reservoir'])
        self.inflow_db = entsoe.Database(db=self.db_path / self.opt_db_files['inflow'])
        self.gen_db = entsoe.Database(db=Path(self.db_path) / self.opt_db_files['gen'])
        self.maf_pecd_db = maf_pecd_data.Database(db=Path(self.db_path) / self.opt_db_files['maf_pecd'])
        self.maf_hydro_db = maf_hydro_data.Database(db=Path(self.db_path) / self.opt_db_files['maf_hydro'])
        self.capacity_db = entsoe.Database(db=Path(self.db_path) / self.opt_db_files['capacity'])

        if self.opt_nucl_individual_units:
            self.db_exists(self.db_path / self.opt_db_files['unit'])
            self.unit_db = entsoe.DatabaseGenUnit(db=Path(self.db_path)/self.opt_db_files['unit'])

        starttime = self.starttime
        endtime = self.endtime
        cet = False

        if prt:
            print('Loading Excel data')
        self.load_shares = pd.read_excel(self.data_path / f'load_shares.xlsx',index_col=0,squeeze=True)
        for a in self.areas:
            if a not in self.load_shares.index:
                self.load_shares.at[a] = 1

        # load generation statistics
        self.stats = pd.read_excel(self.data_path / 'gen_stats.xlsx',header=[0,1],index_col=0,sheet_name=f'{self.opt_capacity_year}')
        # load entsoe capacities
        self.gen_capacity = self.capacity_db.select_capacity_wrap(areas=self.areas,year=self.opt_capacity_year)

        if prt:
            print('Loading GenPerType data')
        # Used to plot generation per type and to complement missing Nordpool data
        # aggregate hydro and thermal generation
        self.entsoe_data = self.gen_db.select_gen_per_type_wrap_v2(starttime=starttime,endtime=endtime,
                                                                      type_map=entsoe_type_map,cet_time=cet,drop_data=False,
                                                                      areas=self.areas,print_output=prt,drop_pc=95)
        if prt:
            print('Loading demand data')
        # demand data
        self.demand = self.load_db.select_load_wrap(starttime=starttime,endtime=endtime,cet_time=cet,areas=self.areas,print_output=prt)

        # reservoir content
        self.reservoir = self.reservoir_db.select_reservoir_wrap(starttime=starttime,endtime=endtime,
                             areas=self.areas,cet_time=cet,normalize=self.opt_reservoir_data_normalized,offset=self.opt_reservoir_offset)

        # Load production data for individual units
        if self.opt_nucl_individual_units:
            self.prod_per_unit,self.units = self.unit_db.select_data(start=starttime,end=endtime,
                                                 countries=[c for c in self.opt_countries if \
                                                            sum([1 for a in country_to_areas[c] if a in self.opt_nucl_individual_units])])

        if prt:
            print('Loading external price data')
        # price data - only needed for external areas with variable transfer
        self.price_external = self.price_db.select_price_data(starttime=starttime,endtime=endtime,cet_time=cet,
                                                           areas=self.opt_set_external_price)

        # get flows for fixed connections
        if prt:
            print('Loading external exchange data')
        self.exchange = self.exchange_db.select_flow_data( \
                connections=list(self.xtrans_ext.loc[self.fixed_transfer_connections,'label_fw']),
                starttime=starttime,
                endtime=endtime,
                table=self.opt_exchange_data_type,
                cet_time=cet,
                area_sep=self.area_sep_str)

        if prt:
            print('Loading exchange capacities')
        # load exchange capacity
        if self.opt_use_var_exchange_cap:
            self.exchange_capacity = self.exchange_db.select_flow_data(table='capacity',area_sep=self.area_sep_str,cet_time=cet,
                                                                       starttime=starttime,endtime=endtime,print_output=prt,
                                                                       connections=list(self.xtrans_int['label_fw'])+list(self.xtrans_int['label_bw']) + \
                                                                                   list(self.xtrans_ext.loc[self.fixed_price_connections,'label_fw'])+ \
                                                                                   list(self.xtrans_ext.loc[self.fixed_price_connections,'label_bw']),
                                                                       drop_na_col=True)


        if prt:
            print('Loading inflow data')
        self.get_inflow_data()

        impute_list = ['reservoir','inflow','demand','price_external','exchange']

        # if self.opt_use_var_exchange_cap:
            # self.impute_capacity_values()
        # interpolate missing values in data
        self.impute_values(impute_list,limit=self.opt_impute_limit,prt=prt)
                        

        # scale up demand
        self.demand = self.demand * self.opt_load_scale

        # replace negative values with zeros
        for name in self.opt_nonnegative_data:
            entity = self.__getattribute__(name)
            entity.clip(0,inplace=True)


    def setup_gen(self):
        prt = self.opt_print['setup']
        if prt:
            print('Setting up generators')
        stats = self.stats
        self.generators_def = generators_def

        # get generator data
        nGen = 0
        gidx = 1
        self.gen_data = pd.DataFrame(index=range(1,nGen+1),
                                     columns=['area','gtype','c2','c1','c0','pmax','pmin','rampup','rampdown'])

        # load cost fit
        with open(self.data_path/f'costfit/{self.opt_costfit_tag}_fit.pkl','rb') as f:
            self.costfit = pickle.load(f)

        for area in self.areas:
            for gtype in self.generators_def[area]:
                if (gtype == 'Hydro' and self.opt_pmax_type_hydro == 'stats') or \
                        (gtype != 'Hydro' and self.opt_pmax_type == 'stats'):
                    pmax = stats.at['max',(area,gtype)]
                else:
                    pmax = self.gen_capacity.at[area,gtype]
                    if np.isnan(pmax): # missing values, use from stats
                        pmax = stats.at['max',(area,gtype)]
                        if prt:
                            print(f'No entso-e capacity value for {area} {gtype}')
                pmin = stats.at['min',(area,gtype)]
                rampup = stats.at['maxramp',(area,gtype)]
                rampdown = stats.at['minramp',(area,gtype)]

                # cost coefficients
                c0 = 0
                if gtype == 'Nuclear':
                    c2 = 0
                    c1 = self.opt_nuclear_cost
                elif gtype == 'Hydro':
                    c2 = 0
                    c1 = 0
                else: # Thermal
                    c2 = self.costfit[area][gtype]['k']/2
                    c1 = self.costfit[area][gtype]['mavg']
                    if self.opt_co2_price is not None and self.opt_co2_price > 0:
                        c1 += self.opt_co2_price*self.opt_co2_intensity
                        if self.opt_co2_price_offset_year is not None:
                            c1 -= co2_price_ets[self.opt_co2_price_offset_year]*self.opt_co2_intensity
                # check if cost parameters are strange, e.g. decreasing marginal cost
                if c2 < 0 or np.isnan(c2):
                    c2 = 0
                    c1 = self.opt_default_thermal_cost
                    if prt:
                        print(f'Using default constant MC costs for {area} {gtype}')

                self.gen_data = self.gen_data.append(pd.DataFrame(columns=self.gen_data.columns,index=[gidx],
                                              data=[[area,gtype,c2,c1,c0,pmax,pmin,rampup,rampdown]]))
                gidx += 1

        self.nGen = gidx - 1
        self.idx_gen = range(1,self.nGen+1)
        self.idx_thermal_gen = [g for g in self.idx_gen if self.gen_data.at[g,'gtype'] == 'Thermal']

        # generators with non-zero marignal cost
        self.idx_cost_gen = [g for g in self.idx_gen if not self.gen_data.at[g,'gtype'] == 'Hydro']

        if self.opt_pmin_zero:
            self.gen_data.loc[:,'pmin'] = 0

        # set maximum nuclear capacity based on this week
        for g in self.gen_data.index:
            if self.gen_data.at[g,'gtype'] == 'Nuclear':
                # note: this is maximum for whole period, is actually not used
                # instead weekly maximum values are used
                self.gen_data.at[g,'pmax'] = self.entsoe_data[self.gen_data.at[g,'area']]['Nuclear'].max()

                # overwrite nuclear cost
                if not self.opt_nuclear_cost is None:
                    self.gen_data.at[g,'c1'] = self.opt_nuclear_cost

                # overwrite nuclear ramp rate
                if not self.opt_nucl_ramp is None:
                    self.gen_data.at[g,'rampup'] = self.gen_data.at[g,'pmax'] * self.opt_nucl_ramp/100
                    self.gen_data.at[g,'rampdown'] = - self.gen_data.at[g,'pmax'] * self.opt_nucl_ramp/100


        def tag_gen_cost():
            pass
        if prt:
            print('Setting up generator variable costs')
        # generator variable costs
        if self.opt_use_var_cost:

            self.gen_c2 = pd.DataFrame(dtype=float,index=self.timerange,columns=self.gen_data.index)
            self.gen_c1 = pd.DataFrame(dtype=float,index=self.timerange,columns=self.gen_data.index)

            for g in self.idx_gen:
                area = self.gen_data.at[g,'area']
                gtype = self.gen_data.at[g,'gtype']

                print_flag_areas = []
                binstart = str_to_date(self.costfit['starttime'])
                binsize = self.costfit['binsize']
                for t in self.idx_time:
                    # it is assumed costs are fitted for correct year
                    # get costs depending on type
                    if gtype == 'Thermal': # variable cost data
                        dt = self.timerange[t]
                        c2 = self.costfit[area][gtype]['k']/2
                        c1 = self.costfit[area]['Thermal']['m'][time_to_bin(
                            dt,binstart=binstart,binsize=binsize)]
                        if self.opt_co2_price is not None and self.opt_co2_price > 0:
                            c1 += self.opt_co2_price*self.opt_co2_intensity
                            if self.opt_co2_price_offset_year is not None:
                                c1 -= co2_price_ets[self.opt_co2_price_offset_year]*self.opt_co2_intensity
                        if self.opt_overwrite_bad_costfits and (c2 < self.opt_c2_min or c2 > self.opt_c2_max):
                            # use default cost
                            c2 = self.gen_data.at[g,'c2']
                            c1 = self.gen_data.at[g,'c1']
                            # show message about overwrite
                            if area not in print_flag_areas:
                                print_flag_areas.append(area)
                                if prt:
                                    print(f'Using constant costs for {area}')
                    else: # use constant costs from gen_data
                        c2 = self.gen_data.at[g,'c2']
                        c1 = self.gen_data.at[g,'c1']
                    self.gen_c2.at[self.timerange[t],g] = c2
                    self.gen_c1.at[self.timerange[t],g] = c1

        # calculate maximum nuclear generation per week
        # USE INDIVIDUAL NUCLEAR GENERATION DATA
        def tag_nuclear():
            pass
        if prt:
            print('Setting up nuclear generation')
        self.nuclear_areas = [a for a in self.areas if 'Nuclear' in self.generators_def[a]]
        for a in self.nuclear_areas:
            if a not in self.opt_nucl_add_cap:
                self.opt_nucl_add_cap[a] = 0.0
        
        #%%
        # variable nuclear limit
        self.nuclear_hourly = pd.DataFrame(dtype=float,index=self.timerange_p1,columns=self.nuclear_areas)
        # fix values for 1 day intervals, the do linear interpolation
        self.nuclear_units = {}
    
        for a in self.nuclear_areas:
            if a in self.opt_nucl_individual_units:
                self.nuclear_units[a] = [idx for idx in self.units.index if self.units.at[idx,'type'] == 'Nuclear'
                                      and self.units.at[idx,'country'] == self.area_to_country[a]
                                      and self.units.at[idx,'name'] not in self.opt_nucl_units_exclude]
                max_rolling = self.prod_per_unit.loc[:,self.nuclear_units[a]].sum(axis=1).rolling(
                    window=168,min_periods=1,center=True).max()
            else:
                max_rolling = self.entsoe_data[a]['Nuclear'].rolling(window=168,min_periods=1,center=True).max()
            for d in self.daysrange_p1:
                self.nuclear_hourly.at[d,a] = max_rolling.at[d] + self.opt_nucl_add_cap[a]
        # interpolate linearly
        self.nuclear_hourly.interpolate(inplace=True)

        # combined generators - define which generator units make up generators with ramp constraints
        # Note: Several units of the same type can be used within an area, e.g. in order to create a piecewise linear
        # cost function for that type and area. Some constraints, e.g. ramp constraints, should then be enforced on
        # the aggregate production of those generators. For this reason there is a set for the "combined generators"
        self.gen_comb = {}
        idx = 1
        for area in self.areas:
            for gtype in self.generators_def[area]:
                # find all generator units which belong to this generator
                units = []
                for gen in self.gen_data.index:
                    if self.gen_data.at[gen,'area'] == area and self.gen_data.at[gen,'gtype'] == gtype:
                        units.append(gen)
                self.gen_comb[idx] = units
                idx += 1
        self.nGenComb = idx-1


        self.gen_data_comb = pd.DataFrame(index=range(1,self.nGenComb+1),columns=['rampup','rampdown'])
        for index in self.gen_data_comb.index:
            self.gen_data_comb.at[index,'rampup'] = self.gen_data.at[self.gen_comb[index][0],'rampup']
            self.gen_data_comb.at[index,'rampdown'] = self.gen_data.at[self.gen_comb[index][0],'rampdown']


        # generators in each area
        self.gen_in_area = {}
        for area in self.areas:
            self.gen_in_area[area] = [g for g in range(1,self.nGen+1) if self.gen_data.at[g,'area'] == area]

    def setup_reserves(self):
        prt = self.opt_print['setup']

        for c in self.opt_countries:
            if c not in self.opt_reserves_fcrn:
                self.opt_reserves_fcrn[c] = 0

        # amount of reserves per area
        self.reserve_data = pd.DataFrame(index=self.areas,columns=['FCR-N','FCR-D','Rp','Rn'])
        for area in self.reserve_data.index:
            if 'SE' in area:
                country = 'SE'
            elif 'NO' in area:
                country = 'NO'
            elif 'DK' in area:
                country = 'DK'
            else:
                country = area
            #all_areas = [a for a in self.areas if country in a]

            gen_hydro = [idx for idx in self.gen_data.index if self.gen_data.at[idx,'gtype'] == 'Hydro']
            gen_area = [idx for idx in self.gen_data.index if self.gen_data.at[idx,'area'] == area]
            gen_country = [idx for idx in self.gen_data.index if country in self.gen_data.at[idx,'area']]
            # allocate in proportion to share of hydro generation
            if self.gen_data.loc[intersection(gen_hydro,gen_country),'pmax'].sum() > 0:
                self.reserve_data.at[area,'FCR-N'] = self.opt_reserves_fcrn[country]* \
                                                     self.gen_data.loc[intersection(gen_hydro,gen_area),'pmax'].sum() / self.gen_data.loc[intersection(gen_hydro,gen_country),'pmax'].sum()
            else:
                self.reserve_data.at[area,'FCR-N'] = 0

            # FCR-D in proportion to FCR-N
            self.reserve_data.at[area,'FCR-D'] = self.reserve_data.at[area,'FCR-N'] * self.opt_reserves_fcrd / np.sum([self.opt_reserves_fcrn[a] for a in self.opt_reserves_fcrn])
            self.reserve_data.at[area,'Rp'] = self.reserve_data.at[area,'FCR-N'] + self.reserve_data.at[area,'FCR-D']
            self.reserve_data.at[area,'Rn'] = self.reserve_data.at[area,'FCR-N']

        # areas with reserves
        self.resareas = [area for area in self.areas if self.reserve_data.at[area,'FCR-N'] > 0]
        # generators providing reserves in each area
        self.reserve_gens = {}
        for area in self.resareas:
            self.reserve_gens[area] = [gen for gen in self.gen_data.index if self.gen_data.at[gen,'area'] == area and self.gen_data.at[gen,'gtype'] == 'Hydro']


        # countries with reserves
        self.rescountries = [c for c in self.opt_reserves_fcrn if self.opt_reserves_fcrn[c] > 0]
        # generators providing reserves in each country
        self.reserve_gens_country = {}
        for c in self.rescountries:
            self.reserve_gens_country[c] = [gen for gen in self.gen_data.index if self.gen_data.at[gen,'area'] in self.country_to_areas[c] and self.gen_data.at[gen,'gtype'] == 'Hydro']

        self.reserve_country_data = pd.DataFrame(index=self.rescountries,columns=self.reserve_data.columns)
        for c in self.reserve_country_data.columns:
            for i in self.reserve_country_data.index:
                self.reserve_country_data.at[i,c] = self.reserve_data.loc[self.country_to_areas[i],c].sum()


    def setup_hydro(self):
        prt = self.opt_print['setup']
        self.reservoir_capacity = self.opt_reservoir_capacity.copy()
        # get missing reservoir capacity as maximum reservoir value
        self.reservoir_max = self.reservoir_db.select_max(table_type='reservoir',areas=self.areas)
        for a in self.hydrores:
            if a not in self.reservoir_capacity or not self.reservoir_capacity[a]:
                self.reservoir_capacity[a] = self.reservoir_max.at[a]
        self.pump_reservoir = self.opt_pump_reservoir.copy()
        # reservoir content in TWh
        for i,val in self.reservoir_capacity.items():
            self.reservoir_capacity[i] = val * GWtoTW
        for i,val in self.pump_reservoir.items():
            self.pump_reservoir[i] = val * GWtoTW


        # One reservoir per area with hydro
        # Mapping from each reservoir to its connected hydro stations
        self.reservoir2hydro = {}
        for area in self.hydrores:
            self.reservoir2hydro[area] = []
            for idx,gen in self.gen_data.iterrows():
                if gen['area'] == area and gen['gtype'] == 'Hydro':
                    self.reservoir2hydro[area].append(idx)

        for a in self.hydrores:
            if a not in self.opt_ror_fraction:
                self.opt_ror_fraction[a] = 0

        # areas with run of river hydro
        self.ror_areas = [a for a in self.areas if a in self.opt_ror_fraction and self.opt_ror_fraction[a] > 0]
        self.ror_countries = [c for c in self.opt_countries if sum([1 for a in self.country_to_areas[c] if a in self.ror_areas])]


        # check which areas have hydro run of river with reserves
        # Note: Run of river hydro is modelled as separate production (like wind and solar),
        # and is not entered within the set of generators. However, when enforcing upward reserve constraints,
        # run of river production still decreases the potential for providing upward reserves from hydro production.
        # Thus
        self.ror_reserve_areas = []
        for a in self.resareas:
            #if a in self.set_HYDRO_AREA:
            if self.area_to_country[a] in self.ror_countries and 'Hydro' in self.generators_def[a]:
                self.ror_reserve_areas.append(a)

        self.ror_reserve_countries = []
        for c in self.rescountries:
            if c in self.ror_countries:
                self.ror_reserve_countries.append(c)

        # HYDRO GENERATORS
        # store data specific to hydro reservoirs in self.hydro_data
        self.hydro_data = pd.DataFrame(index=self.hydrores,columns=['reservoir','waterval'])
        # reservoir capacity
        for area in self.hydro_data.index:
            self.hydro_data.at[area,'reservoir'] = self.reservoir_capacity[area]

        # PUMP HYDRO
        # areas with pumping
        self.pump_areas = [
            a for a in self.hydrores if a in self.opt_pump_capacity and self.opt_pump_capacity[a] > 0
        ]
        # areas with separate reservoir for pumping
        self.pump_res_areas = [a for a in self.pump_areas \
                               if a in self.opt_pump_reservoir and self.opt_pump_reservoir[a] > 0]
        # areas with pumping in inflow reservoir
        self.pump_nores_areas = [a for a in self.pump_areas if a not in self.pump_res_areas]

    def setup_run_of_river(self):
        prt = self.opt_print['setup']
        self.ror_hourly = pd.DataFrame(dtype=float,index=self.timerange,columns=self.ror_areas)
        for area in self.ror_areas:
                self.ror_hourly[area] = self.opt_ror_fraction[area] * \
                                        self.inflow_hourly.loc[self.timerange,area] * GWtoMW
        if self.opt_hydro_daily:
            self.ror_daily = self.ror_hourly.resample('D').sum()

    def setup_inflow_feasibility(self):
        for area in self.hydrores:
            hgens = [g for g in self.idx_gen if self.gen_data.at[g,'area'] == area and self.gen_data.at[g,'gtype'] == 'Hydro']
            pmin = self.gen_data.loc[hgens,'pmin'].sum()
            if area in self.ror_areas:
                minprod = np.array(self.ror_hourly[area])
                minprod[minprod <= pmin] = pmin
                minprod_tot = np.sum(minprod)*MWtoGW
            else:
                minprod_tot = pmin*self.nPeriods*MWtoGW
            # hydro may also have to keep negative reserves
            if self.opt_use_reserves and not self.opt_country_reserves:
                # TODO: Factor 1 should be enough??
                minprod_tot += self.reserve_data.at[area,'Rn']*MWtoGW*self.nPeriods*2
            res_incr = TWtoGW*(self.reservoir_fix.at[self.reservoir_fix.index[1],area] - self.reservoir_fix.at[self.reservoir_fix.index[0],area])
            inflow_tot = self.inflow_hourly[area].sum()
            if minprod_tot + res_incr > inflow_tot:
                incr_val = (minprod_tot + res_incr - inflow_tot)*1.01/self.nPeriods
                print(f'WARNING: Total inflow for {area} cannot satisfy minimum production and start/end reservoir values'
                      + f'\nIncreasing inflow by {incr_val:0.3f} GWh/h to avoid infeasibility!')
                self.inflow_hourly[area] = self.inflow_hourly[area] + incr_val
                if self.opt_hydro_daily:
                    self.inflow_daily[area] = self.inflow_daily[area] + incr_val*24

    def setup_wind(self):
        prt = self.opt_print['setup']

        # wind data
        if self.opt_use_maf_pecd:
            self.wind_areas = [a for a in self.areas if self.opt_wind_capacity_onsh[a] or self.opt_wind_capacity_offsh[a]]

            self.maf_pecd_onsh_areas = list(set([bidz2maf_pecd[a] for a in self.wind_areas if self.opt_wind_capacity_onsh[a]]))
            self.maf_pecd_offsh_areas = list(set([bidz2maf_pecd[a] for a in self.wind_areas if self.opt_wind_capacity_offsh[a]]))

            #% get maf wind data
            self.onsh_maf = self.maf_pecd_db.select_pecd_data(starttime=self.starttime2,endtime=self.endtime2,data_type='onshore',get_areas=self.maf_pecd_onsh_areas)
            self.offsh_maf = self.maf_pecd_db.select_pecd_data(starttime=self.starttime2,endtime=self.endtime2,data_type='offshore',get_areas=self.maf_pecd_offsh_areas)
            # scale with capacity, add offshore and onshore
            self.wind_maf_raw = pd.DataFrame(0.0,index=self.onsh_maf.index,columns=self.wind_areas)
            for a in self.wind_maf_raw.columns:
                ma = bidz2maf_pecd[a]
                if self.opt_wind_capacity_onsh[a]:
                    self.wind_maf_raw[a] += self.onsh_maf[ma]*self.opt_wind_capacity_onsh[a]
                if self.opt_wind_capacity_offsh[a]:
                    self.wind_maf_raw[a] += self.offsh_maf[ma]*self.opt_wind_capacity_offsh[a]

            self.wind = self.copy_data_to_model_year(self.wind_maf_raw)*MWtoGW

        else: # use Entso-e data
            self.wind_areas = [a for a in self.areas if 'Wind' in self.entsoe_data[a]]
            self.wind = pd.DataFrame(index=self.entsoe_data['SE3'].index,columns=self.wind_areas)
            for a in self.wind_areas:
                self.wind[a] = self.entsoe_data[a]['Wind'] * MWtoGW

        self.impute_values(['wind'],limit=self.opt_impute_limit,prt=prt)
        for a in self.wind_areas:
            if a not in self.opt_wind_scale_factor:
                self.opt_wind_scale_factor[a] = 1


    def setup_solar(self):
        prt = self.opt_print['setup']
        if prt:
            print('Setting up solar generation')
        self.solar_capacity = {}
        for c in ['SE','DK','NO','FI']:
            for a in country_to_areas[c]:
                self.solar_capacity[a] = self.load_shares.at[a]*self.opt_solar_cap_by_country[c]*MWtoGW

        # adjust solar capacities with given values per area
        for a in self.opt_solar_cap_by_area:
            self.solar_capacity[a] = self.opt_solar_cap_by_area[a]*MWtoGW

        if self.opt_use_maf_pecd:
            self.solar_areas = [a for a in self.solar_capacity if self.solar_capacity[a] > 0 and a in self.areas]
            self.maf_pecd_solar_areas = list(set([bidz2maf_pecd[a] for a in self.solar_areas]))
            self.solar_maf_raw = self.maf_pecd_db.select_pecd_data(starttime=self.starttime2,endtime=self.endtime2,data_type='pv',get_areas=self.maf_pecd_solar_areas)
            self.solar_maf_mapped = pd.DataFrame(dtype=float,columns=self.solar_areas,index=self.solar_maf_raw.index)
            for a in self.solar_areas:
                self.solar_maf_mapped[a] = self.solar_maf_raw[bidz2maf_pecd[a]]*self.solar_capacity[a]
            self.solar = self.copy_data_to_model_year(self.solar_maf_mapped)
        else:
            self.solar_areas = [a for a in self.areas if 'Solar' in self.entsoe_data[a].columns]
            # use entsoe data
            self.solar = pd.DataFrame(0.0,index=self.timerange_p1,columns=self.solar_areas)
            for a in self.solar_areas:
                self.solar[a] = self.entsoe_data[a]['Solar']*MWtoGW

        self.impute_values(['solar'],limit=self.opt_impute_limit,prt=prt)
    def setup_transmission(self):
        prt = self.opt_print['setup']
        if prt:
            print('Setting up transmission capacity')
        internal_connections = []
        external_connections = []
        for row in nordpool_capacities.iterrows():
            if row[1]['from'] in self.areas and row[1]['to'] in self.areas:
                internal_connections.append(row[0])
            elif row[1]['from'] in self.areas or row[1]['to'] in self.areas:
                external_connections.append(row[0])
        self.xtrans_int = nordpool_capacities.loc[internal_connections,:]
        self.xtrans_ext = nordpool_capacities.loc[external_connections,:]
        self.xtrans_int['c1_change'] = 0
        self.xtrans_int['c2_change'] = 0
        self.xtrans_ext['c1_change'] = 0
        self.xtrans_ext['c2_change'] = 0

        # add future transmission capacity
        if not self.opt_exchange_cap_year is None:
            #update_transfer_capacity(self,year=self.opt_exchange_cap_year)
            xtrans_int_new,xtrans_ext_new = self.update_transfer_capacity(year=self.opt_exchange_cap_year)
            # Note: c1_change, c2_change are treated as firm capacities when using variable capacity limits
            self.xtrans_int = xtrans_int_new
            self.xtrans_ext = xtrans_ext_new

            # internal connections
        self.nXint = self.xtrans_int.__len__()
        self.idx_xint = range(1,self.nXint+1)
        self.xtrans_int.index = self.idx_xint

        # external connections
        self.nXext = self.xtrans_ext.__len__()
        self.idx_xext = range(1,self.nXext+1)
        self.xtrans_ext.index = self.idx_xext

        # internal forward connections for each area
        self.xintf = {}
        for area in self.areas:
            self.xintf[area] = [c for c in range(1,self.nXint+1) if self.xtrans_int.at[c,'from'] == area]
        # internal reverse connections for each area
        self.xintr = {}
        for area in self.areas:
            self.xintr[area] = [c for c in range(1,self.nXint+1) if self.xtrans_int.at[c,'to'] == area]

        ## SETUP EXTERNAL CONNECTIONS ##
        # external connections -> parameter
        # make sure from area is always internal area
        # for row in self.xtrans_ext.iterrows():
        #     if row[1]['from'] not in self.areas:
        #         self.xtrans_ext.at[row[1],'from'] = row[1]['to']
        #         self.xtrans_ext.at[row[1],'to'] = row[1]['from']

        # hvdc ramping rates for external hvdc connections
        for c in self.xtrans_ext.index:
            if self.xtrans_ext.at[c,'to'] == 'DE' and self.xtrans_ext.at[c,'from'] == 'DK1':
                self.xtrans_ext.at[c,'ramp'] = 1e6 # = inf
            else:
                self.xtrans_ext.at[c,'ramp'] = self.opt_hvdc_max_ramp


        ## SETUPT EXTERNAL CONNECTIONS ##
        prt = self.opt_print['setup']
        if prt:
            print('Setting up external connections')

        self.fixed_price_connections = [idx for idx in self.xtrans_ext.index if self.xtrans_ext.at[idx,'to'] in self.opt_set_external_price]
        self.fixed_transfer_connections = [idx for idx in self.xtrans_ext.index if self.xtrans_ext.at[idx,'to'] not in self.opt_set_external_price]

        # Create sets mapping areas to transfer connections
        self.xext_ft = {}
        self.xext_fp = {}

        for area in self.areas:
            self.xext_ft[area] = [c for c in range(1,self.nXext+1) if self.xtrans_ext.at[c,'from'] == area and c in self.fixed_transfer_connections]
            self.xext_fp[area] = [c for c in range(1,self.nXext+1) if self.xtrans_ext.at[c,'from'] == area and c in self.fixed_price_connections]

        self.area_sep_str = '->' # string separating areas in labels for connections
        self.xtrans_ext['label_fw'] = ''
        self.xtrans_ext['label_bw'] = ''
        for i in self.xtrans_ext.index:
            self.xtrans_ext.at[i,'label_fw'] = f"{self.xtrans_ext.at[i,'from']}{self.area_sep_str}{self.xtrans_ext.at[i,'to']}"
            self.xtrans_ext.at[i,'label_bw'] = f"{self.xtrans_ext.at[i,'to']}{self.area_sep_str}{self.xtrans_ext.at[i,'from']}"
        self.xtrans_int['label_fw'] = ''
        self.xtrans_int['label_bw'] = ''
        for i in self.xtrans_int.index:
            self.xtrans_int.at[i,'label_fw'] = f"{self.xtrans_int.at[i,'from']}{self.area_sep_str}{self.xtrans_int.at[i,'to']}"
            self.xtrans_int.at[i,'label_bw'] = f"{self.xtrans_int.at[i,'to']}{self.area_sep_str}{self.xtrans_int.at[i,'from']}"

        ## HVDC RAMPING ##

        # hvdc ramping restrictions - maximum 600 MW/hour, also joint restriction for konti-skan and skagerak

        #Konti-Skan    SE3 <-> DK1 - int 10
        #Skagerrak    NO2 <-> DK1 - int 19
        #NorNed    NO2 <-> NL - ext
        #Kontek    GER <-> DK2 -ext
        #SwePol    SE4 <-> PL - ext
        #Baltic Cable    SE4 <-> GER - ext
        #Estlink    FI <-> EE - ext
        #Storebelt    DK1 <-> DK2 - int 18
        #NordBalt    LT <-> SE4 - ext
        #LitPol    PL <-> LT - ext

        self.combined_hvdc = {
            1:[10,19],
            2:[10],
            3:[19],
            4:[18],
        }


    def update_transfer_capacity(self,year=2035,tc_table=new_trans_cap):
        """ Update transfer capacity using future expansion plans, considering
        all new capacity until given year """
    
        xtrans_int = pd.DataFrame(data=0,index=self.xtrans_int.index,columns=['from','to','c1','c2','c1_change','c2_change'])
        xtrans_ext = pd.DataFrame(data=0,index=self.xtrans_ext.index,columns=['from','to','c1','c2','c1_change','c2_change'])
        xtrans_int.loc[:,['from','to','c1','c2']] = self.xtrans_int.loc[:,['from','to','c1','c2']]
        xtrans_ext.loc[:,['from','to','c1','c2']] = self.xtrans_ext.loc[:,['from','to','c1','c2']]
    
        # drop ramp rate column (shouldn't exist unless function is called after setup)    
        if 'ramp' in xtrans_ext.columns:
            xtrans_ext = xtrans_ext.drop(columns=['ramp'])
    
    
        for cidx in tc_table.index:
            if tc_table.at[cidx,'year'] <= year:
                # add capacity
                # check if it's external or internal connection
                if tc_table.at[cidx,'from'] in self.areas and tc_table.at[cidx,'to'] in self.areas:
                    # internal connection
                    # find connection 
                    new_conn = []
                    found_conn = False
                    for idx in xtrans_int.index:
                        if xtrans_int.at[idx,'from'] == tc_table.at[cidx,'from'] and xtrans_int.at[idx,'to'] == tc_table.at[cidx,'to']:
                            # add to existing connection
                            xtrans_int.at[idx,'c1_change'] += tc_table.at[cidx,'c1_change']
                            xtrans_int.at[idx,'c2_change'] += tc_table.at[cidx,'c2_change']
                            found_conn = True
                            break
                        elif xtrans_int.at[idx,'from'] == tc_table.at[cidx,'to'] and xtrans_int.at[idx,'to'] == tc_table.at[cidx,'from']:
                            # add to existing connection, reverse direction    
                            xtrans_int.at[idx,'c1_change'] += tc_table.at[cidx,'c2_change']
                            xtrans_int.at[idx,'c2_change'] += tc_table.at[cidx,'c1_change']
                            found_conn = True
                            break
                    if not found_conn:
                        # add new internal connection
                        if tc_table.at[cidx,'from'] in self.areas:
                            new_conn.append([tc_table.at[cidx,'from'],tc_table.at[cidx,'to'],0,0,tc_table.at[cidx,'c1_change'],tc_table.at[cidx,'c2_change'],])
                        else:
                            new_conn.append([tc_table.at[cidx,'to'],tc_table.at[cidx,'from'],0,0,tc_table.at[cidx,'c2_change'],tc_table.at[cidx,'c1_change'],])
                        xtrans_int = xtrans_int.append(pd.DataFrame(new_conn,columns=['from','to','c1','c2','c1_change','c2_change']),ignore_index=True)
                else:
                    # external connection
                    # find connection
                    new_conn = []
                    found_conn = False
                    for idx in xtrans_ext.index:
                        if xtrans_ext.at[idx,'from'] == tc_table.at[cidx,'from'] and xtrans_ext.at[idx,'to'] == tc_table.at[cidx,'to']:
                            # add to existing connection
                            xtrans_ext.at[idx,'c1_change'] += tc_table.at[cidx,'c1_change']
                            xtrans_ext.at[idx,'c2_change'] += tc_table.at[cidx,'c2_change']
                            found_conn = True
                            break
                        elif xtrans_ext.at[idx,'from'] ==  tc_table.at[cidx,'to'] and xtrans_ext.at[idx,'to'] == tc_table.at[cidx,'from']:
                            # add to existing connection, reverse direction
                            xtrans_ext.at[idx,'c1_change'] += tc_table.at[cidx,'c2_change']
                            xtrans_ext.at[idx,'c2_change'] += tc_table.at[cidx,'c1_change']
                            found_conn = True
                            break
                    if not found_conn:
                        # add new external connection
                        if tc_table.at[cidx,'from'] in self.areas:
                            new_conn.append([tc_table.at[cidx,'from'],tc_table.at[cidx,'to'],0,0,tc_table.at[cidx,'c1_change'],tc_table.at[cidx,'c2_change']])
                        else:
                            new_conn.append([tc_table.at[cidx,'to'],tc_table.at[cidx,'from'],0,0,tc_table.at[cidx,'c2_change'],tc_table.at[cidx,'c1_change']])
                        xtrans_ext = xtrans_ext.append(pd.DataFrame(new_conn,columns=['from','to','c1','c2','c1_change','c2_change']),ignore_index=True)
    
    
        return xtrans_int,xtrans_ext
    
    def setup_inflow(self):
        prt = self.opt_print['setup']

        if prt:
            print('Interpolate_inflow')
        self.interpolate_inflow()

        if self.opt_use_maf_inflow:
            self.add_hourly_maf_inflow()

        if self.opt_hydro_daily:
            # calculate daily inflow
            self.inflow_daily = self.inflow_hourly.resample('D').sum()

    def setup_bounds(self,lowres=False):
        """ Create time-varying bounds for variables
        Bounds are in the form of dicts with set values as indices, for setup of problem
        using gurobi (or pyomo)
        bound = {(i1,i2,...):value for i1 in set1.. }
        """
        prt = self.opt_print['setup']
        if lowres:
            idx_time = range(self.nPeriods_lr)
            timerange = self.timerange_lr
            if self.opt_use_var_exchange_cap:
                exchange_capacity = self.exchange_capacity_lr
            solar = self.solar_lr
            wind = self.wind_lr
            demand = self.demand_lr
            ror_hourly = self.ror_hourly_lr
        else:
            idx_time = self.idx_time
            timerange = self.timerange
            if self.opt_use_var_exchange_cap:
                exchange_capacity = self.exchange_capacity
            solar = self.solar
            wind = self.wind
            demand = self.demand
            ror_hourly = self.ror_hourly
            
        max_lims = {}
        min_lims = {}

        # limit for external variable connections
        max_XEXT = {}
        min_XEXT = {}

        for c in self.fixed_price_connections:
            c1 = self.xtrans_ext.at[c,'label_fw']
            c2 = self.xtrans_ext.at[c,'label_bw']
            a1 = c1.split(self.area_sep_str)[0]
            a2 = c2.split(self.area_sep_str)[0]
            if not self.opt_use_var_exchange_cap or c1 not in exchange_capacity.columns \
                    or (a1,a2) in self.opt_nominal_capacity_connections \
                    or (a2,a1) in self.opt_nominal_capacity_connections:
                if self.opt_use_var_exchange_cap:
                    if prt:
                        print(f"Using fixed transfer capacity for {c1}")
                for t in idx_time:
                    max_XEXT[(c,t)] = MWtoGW * self.xtrans_ext.loc[c,['c1','c1_change']].sum()
            else:
                for t in idx_time:
                    max_XEXT[(c,t)] = MWtoGW * max(self.opt_min_exchange_cap,
                        self.xtrans_ext.at[c,'c1_change']+exchange_capacity.at[timerange[t],c1])

            if not self.opt_use_var_exchange_cap or c2 not in exchange_capacity.columns:
                if self.opt_use_var_exchange_cap:
                    if prt:
                        print(f"Using fixed transfer capacity for {c1}")
                for t in idx_time:
                    min_XEXT[(c,t)] = -MWtoGW * self.xtrans_ext.loc[c,['c2','c2_change']].sum()
            else:
                for t in idx_time:
                    min_XEXT[(c,t)] = -MWtoGW * max(self.opt_min_exchange_cap,
                        self.xtrans_ext.at[c,'c2_change']+exchange_capacity.at[timerange[t],c2])
        max_lims['XEXT'] = max_XEXT
        min_lims['XEXT'] = min_XEXT

        # limit for internal connections
        max_X1 = {}
        max_X2 = {}
        for c in self.idx_xint:
            c1 = self.xtrans_int.at[c,'label_fw']
            c2 = self.xtrans_int.at[c,'label_bw']
            a1 = c1.split(self.area_sep_str)[0]
            a2 = c2.split(self.area_sep_str)[0]
            if not self.opt_use_var_exchange_cap or c1 not in exchange_capacity.columns \
                    or (a1,a2) in self.opt_nominal_capacity_connections \
                    or (a2,a1) in self.opt_nominal_capacity_connections:
                if self.opt_use_var_exchange_cap:
                    if prt:
                        print(f"Using fixed transfer capacity for {c1}")
                for t in idx_time:
                    max_X1[(c,t)] = MWtoGW * self.xtrans_int.loc[c,['c1','c1_change']].sum()
            else: # variable capacity
                for t in idx_time:
                    max_X1[(c,t)] = MWtoGW * max(self.opt_min_exchange_cap,
                        self.xtrans_int.at[c,'c1_change']+exchange_capacity.at[timerange[t],c1])
            if not self.opt_use_var_exchange_cap or c2 not in exchange_capacity.columns \
                    or (a1,a2) in self.opt_nominal_capacity_connections \
                    or (a2,a1) in self.opt_nominal_capacity_connections:
                if self.opt_use_var_exchange_cap:
                    if prt:
                        print(f"Using fixed transfer capacity for {c2}")
                for t in idx_time:
                    max_X2[(c,t)] = MWtoGW * self.xtrans_int.loc[c,['c2','c2_change']].sum()
            else:
                for t in idx_time:
                    max_X2[(c,t)] = MWtoGW * max(self.opt_min_exchange_cap,
                        self.xtrans_int.at[c,'c2_change']+exchange_capacity.at[timerange[t],c2])
        max_lims['X1'] = max_X1
        max_lims['X2'] = max_X2

        max_lims['SOLAR'] = {
            (a,t):solar.at[timerange[t],a] for a in self.solar_areas for t in idx_time
        }

        max_lims['WIND'] = {
            (a,t):wind.at[timerange[t],a]*self.opt_wind_scale_factor[a]
            for a in self.wind_areas for t in idx_time
        }

        max_LS = {}
        for area in self.areas:
            for t in idx_time:
                max_LS[(area,t)] = demand.at[timerange[t],area]*MWtoGW
        max_lims['LS'] = max_LS

        max_lims['HROR'] = {(a,t):self.ror_hourly.at[timerange[t],a]*MWtoGW for a in self.ror_areas for t in idx_time}

        # generator limits
        max_PG = {}
        min_PG = {}
        for g in self.idx_gen:
            gtype = self.gen_data.at[g,'gtype']
            area = self.gen_data.at[g,'area']
            for t in idx_time:
                time = timerange[t]
                if gtype == 'Nuclear':
                    pmax = self.nuclear_hourly.at[time,area]
                    pmin = self.opt_nucl_min_lvl * pmax
                elif gtype == 'Hydro':
                    pmax = self.gen_data.at[g,'pmax']
                    pmin = 0
                else:
                    pmax = self.gen_data.at[g,'pmax']
                    pmin = self.gen_data.at[g,'pmin']

                max_PG[(g,t)] = pmax*MWtoGW
                min_PG[(g,t)] = pmin*MWtoGW
        max_lims['PG'] = max_PG
        min_lims['PG'] = min_PG

        max_RES = {}
        for a in self.hydrores:
            if not self.opt_hydro_daily or lowres:
                for t in idx_time:
                    max_RES[(a,t)] = self.reservoir_capacity[a]
            else:
                for t in self.idx_day:
                    max_RES[(a,t)] = self.reservoir_capacity[a]
        max_lims['RES'] = max_RES

        max_lims['PUMP'] = {
            (a,t):self.opt_pump_capacity[a]*MWtoGW for a in self.pump_areas for t in self.idx_time
        }
        max_lims['PRES'] = {
            (a,t):self.pump_reservoir[a] for a in self.pump_res_areas for t in self.idx_time
        }

        # round small values to zero
        for name in self.opt_bound_cut:
            thrs = self.opt_bound_cut[name]
            round = False
            count = 0
            minmax = name.split('_')[0]
            var = name.split('_')[1]
            if minmax == 'max' and var in max_lims:
                entity = max_lims[var]
                round = True
            elif minmax == 'min' and var in min_lims:
                entity = min_lims[var]
                round = True
            if round:
                for i,val in entity.items():
                    if val < thrs and val > 0:
                        entity[i] = 0
                        count += 1
                if prt:
                    print(f'Rounded {count} values to zero in {name}')

        if not lowres:
            for name in max_lims:
                self.__setattr__(f'max_{name}',max_lims[name])
            for name in min_lims:
                self.__setattr__(f'min_{name}',min_lims[name])
        else:
            for name in max_lims:
                self.__setattr__(f'max_{name}_LR',max_lims[name])
            for name in min_lims:
                self.__setattr__(f'min_{name}_LR',min_lims[name])

    def get_df_bounds(self):
        prt = self.opt_print['postprocess']

        """
        For variables in vars_df_up_bound also make data-frames with upper bounds, useful for doing calculations

        e.g. up_SOLAR = pd.DataFrame(index=timerange,columns=solar_areas)
        """

        for var in self.vars_df_up_bound:
            if self.vars_df_up_bound[var][-1] == 'idx_time' and self.vars_df_up_bound[var].__len__() == 2:
                cols = self.__getattribute__(self.vars_df_up_bound[var][0])
                df = pd.DataFrame([[self.__getattribute__(f'max_{var}')[(c,t)] for c in cols] for t in self.idx_time],
                                  columns=cols,index=self.timerange)
                self.__setattr__(f'up_{var}',df)
            else:
                print(f'Get upper bound for variable {var} not implemented!')

    def get_results_from_child(self):
        """ Get result variables, duals, and bounds from optimization problem """
        prt = self.opt_print['postprocess']
        self.res_obj = self.cm.get_objective_value()

        # read results into dataframes
        if prt:
            print('Reading results into Panda data frames')

        for v in self.pp_opt.get_vars:
            entity = self.cm.get_variable(f'var_{v}')
            # convert to date index
            if self.opt_hydro_daily and v in self.pp_opt.daily_vars:
                entity.index = self.daysrange
            else:
                entity.index = self.timerange
            setattr(self,f'res_{v}',entity)


        # increment time index of instantaneous variables
        for var in [v for v in self.pp_opt.inst_vars if v in self.pp_opt.get_vars]:
            entity = getattr(self,f'res_{var}')
            if self.opt_hydro_daily and var in self.pp_opt.daily_vars:
                entity.index += datetime.timedelta(days=1)
            else:
                entity.index += datetime.timedelta(hours=1)

        # get dual variables
        if prt:
            print('Getting dual variables')
        for v in self.pp_opt.get_duals:
            constr = f'constr_{v}'
            if hasattr(self.cm,constr):
                entity = self.cm.get_dual(constr)
                # convert to date index
                if v not in ['FIX_RESERVOIR']:
                    if entity.index.__len__() == self.nPeriods:
                        entity.index = self.timerange
                    elif entity.index.__len__() == self.nDays:
                        entity.index = self.daysrange
                setattr(self,f'dual_{constr}',entity)

    def get_rmse_data(self):
        prt = self.opt_print['postprocess']
        # get data used for rmse calculations

        # production data
        starttime = self.starttime
        endtime = (str_to_date(self.endtime)+datetime.timedelta(hours=-1)).strftime('%Y%m%d:%H')

        if self.fopt_plots['transfer_internal'] or self.fopt_plots['transfer_external'] or self.fopt_calc_rmse['transfer']:
            # get transfer data, for internal and external variable connections
            self.df_exchange_rmse = self.exchange_db.select_flow_data(
                connections=[self.xtrans_ext.at[i,'label_fw'] for i in self.fixed_price_connections] \
                    + [f"{self.xtrans_int.at[i,'from']}{self.area_sep_str}{self.xtrans_int.at[i,'to']}"
                       for i in self.xtrans_int.index],
                starttime=starttime,
                endtime=endtime,
                table=self.fopt_rmse_transfer_data_type,
                cet_time=True)
        if self.fopt_calc_rmse['price'] or self.fopt_plots['price']:
            # get interal price data
            self.df_price_internal = self.price_db.select_price_data(starttime=starttime,
                                                                  endtime=endtime,
                                                                  cet_time=True,
                                                                  areas=self.areas)
            self.impute_values(['df_price_internal'],limit=self.opt_impute_limit,prt=prt)

    def post_process_calculations(self):
        prt = self.opt_print['postprocess']
        ## CALCULATE CURTAILMENT ##
        for v in self.pp_opt.get_cur_vars:
            df = self.__getattribute__(f'up_{v}') - self.__getattribute__(f'res_{v}')
            self.__setattr__(f'res_cur_{v}',df)

        ## CALCULATE DEMAND ##
        self.res_D = self.__getattribute__(f'up_LS') - self.__getattribute__('res_LS')

        ## CALCULATE COSTS ##
        # cost of load shedding
        self.res_loadshed_cost = MWtoGW*self.res_LS.sum().sum()
        # cost of thermal generation
        self.res_thermal_cost = MWtoGW*sum( sum( self.gen_data.at[g,'c2']*self.res_PG.at[t,g]**2/MWtoGW + self.gen_data.at[g,'c1']*self.res_PG.at[t,g] for t in self.timerange) for g in self.idx_cost_gen)
        # net sales of power
        self.res_exp_rev = MWtoGW*sum( sum(self.price_external.at[self.timerange[t],self.xtrans_ext.at[x,'to']]*self.res_XEXT.at[self.timerange[t],x] for t in self.idx_time) for x in self.fixed_price_connections)
        # value of stored water
        self.res_water_value =  MWtoGW*sum( self.hydro_data.at[h,'waterval']*self.res_RES.at[self.res_RES.index[-1],h] for h in self.hydrores )

        ## CALCULATE GENERATION ##
        self.res_gen =  {}
        for area in self.areas:
            self.res_gen[area] = pd.DataFrame(index=self.timerange,columns=self.generators_def[area])
            for gtype in self.generators_def[area]:
                self.res_gen[area].loc[:,gtype] = self.res_PG.loc[:,
                                                     [gidx for gidx in self.idx_gen if
                                                      self.gen_data.at[gidx,'area']==area and
                                                      self.gen_data.at[gidx,'gtype'] == gtype]].sum(axis=1)
        # get wind production
        for area in self.res_WIND.columns:
            self.res_gen[area].loc[:,'Wind'] = self.res_WIND.loc[:,area]
        # get solar production
        for area in self.res_SOLAR.columns:
            self.res_gen[area].loc[:,'Solar'] = self.res_SOLAR.loc[:,area]
            # get hydro ror
        for area in self.res_HROR.columns:
            self.res_gen[area].loc[:,'HROR'] = self.res_HROR.loc[:,area]
        for area in self.pump_res_areas:
            self.res_gen[area].loc[:,'REL'] = self.res_REL.loc[:,area]

        ## CALCULATE TRANSFERS ##
        # get internal transmissions
        self.res_xint = pd.DataFrame(dtype=float,index=self.timerange,columns=self.idx_xint)
        for conn in self.idx_xint:
            self.res_xint[conn] = self.res_X1.loc[:,conn] - self.res_X2.loc[:,conn]


        # get external transmissions
        self.res_xext = {}
        for conn in self.idx_xext:
            self.res_xext[conn] = pd.Series(index=self.timerange)
            if conn in self.fixed_transfer_connections:
                for t in self.idx_time:
                    self.res_xext[conn].at[self.timerange[t]] = MWtoGW * \
                        self.exchange.at[self.timerange[t],self.xtrans_ext.at[conn,'label_fw']]

            else:
                self.res_xext[conn] = self.res_XEXT.loc[:,conn]

        def goto1():
            pass
        # get net exports for each area
        self.res_exp = {}
        for area in self.areas:
            self.res_exp[area] = pd.DataFrame(0.0,index=self.timerange,columns=['Exports int','Exports ext'])
        for area in self.areas:

            for conn in self.xext_ft[area]:
                self.res_exp[area].loc[:,'Exports ext'] = self.res_exp[area].loc[:,'Exports ext'] + self.res_xext[conn]
            for conn in self.xext_fp[area]:
                self.res_exp[area].loc[:,'Exports ext'] = self.res_exp[area].loc[:,'Exports ext'] + self.res_xext[conn]

            # exports to internal regions
            for conn in self.xintf[area]:
                self.res_exp[area].loc[:,'Exports int'] = self.res_exp[area].loc[:,'Exports int'] \
                                                             + self.res_X1.loc[:,conn] - (1-self.opt_loss)*self.res_X2.loc[:,conn]
                # net export = export - (1-loss)*import
            for conn in self.xintr[area]:
                self.res_exp[area].loc[:,'Exports int'] = self.res_exp[area].loc[:,'Exports int'] \
                                                             + self.res_X2.loc[:,conn] - (1-self.opt_loss)*self.res_X1.loc[:,conn]


        ## AGGREGATE GENERATION BY COUNTRIES
        for c in self.multi_area_countries:
            columns = []
            all_columns = ['Nuclear','Thermal','Hydro','Wind','Solar','HROR','REL']
            #col_idx = []
            # get columns for this country
            for area in self.areas:
                if c in area:
                    for gtype in self.res_gen[area].columns:
                        if gtype not in columns:
                            columns.append(gtype)

            self.res_gen[c] = pd.DataFrame(0.0,index=self.timerange,columns=[c for c in all_columns if c in columns])
            for area in self.areas:
                if c in area:
                    for gtype in self.res_gen[area].columns:
                        self.res_gen[c].loc[:,gtype] = self.res_gen[c].loc[:,gtype] + self.res_gen[area].loc[:,gtype]

        # AGGREGATE EXPORTS
        for c in self.multi_area_countries:
            self.res_exp[c] = pd.DataFrame(0,index = self.timerange,columns=['Exports ext','Exports int'])
            for area in self.areas:
                if c in area:
                    self.res_exp[c].loc[:,'Exports int'] = self.res_exp[c].loc[:,'Exports int'] + self.res_exp[area].loc[:,'Exports int']
                    self.res_exp[c].loc[:,'Exports ext'] = self.res_exp[c].loc[:,'Exports ext'] + self.res_exp[area].loc[:,'Exports ext']


        ## TOTAL NET EXPORTS PER AREA (INCLUDING EXTERNAL REGIONS)
        # get total transfer on each connection
        self.res_xext_tot = pd.Series(index=self.xtrans_ext.index)
        self.res_xint_tot = pd.Series(index=self.xtrans_int.index)
        for idx in self.res_xext_tot.index:
            self.res_xext_tot.at[idx] = self.res_xext[idx].sum()
        for idx in self.res_xint_tot.index:
            self.res_xint_tot.at[idx] = self.res_xint[idx].sum()

        # for each area, internal and external, get net exports
        ext_areas = list(set(self.xtrans_ext.loc[:,'to']))
        self.res_net_exports = pd.Series(0,index = self.areas + ext_areas,dtype=float)

        # loop over all connections
        for idx in self.res_xint_tot.index:
            self.res_net_exports.at[ self.xtrans_int.at[idx,'from'] ] = self.res_net_exports.at[ self.xtrans_int.at[idx,'from'] ] + self.res_xint_tot.at[idx]
            self.res_net_exports.at[ self.xtrans_int.at[idx,'to'] ] = self.res_net_exports.at[ self.xtrans_int.at[idx,'to'] ] - self.res_xint_tot.at[idx]
        for idx in self.res_xext_tot.index:
            self.res_net_exports.at[ self.xtrans_ext.at[idx,'from'] ] = self.res_net_exports.at[ self.xtrans_ext.at[idx,'from'] ] + self.res_xext_tot.at[idx]
            self.res_net_exports.at[ self.xtrans_ext.at[idx,'to'] ] = self.res_net_exports.at[ self.xtrans_ext.at[idx,'to'] ] - self.res_xext_tot.at[idx]

        ## RESIDUALS ##

        # check production = demand + exports
        for area in self.areas:
            if area in self.pump_areas:
                self.res_residuals[area] = max(np.abs(self.res_gen[area].sum(axis=1) - self.res_D.loc[:,area] - self.res_exp[area].sum(axis=1) \
                                               - self.res_PUMP[area]))
            else:
                self.res_residuals[area] = max(np.abs(self.res_gen[area].sum(axis=1) - self.res_D.loc[:,area] - self.res_exp[area].sum(axis=1)))

        ## CALCULATE LOSSES AS FRACTION OF INTERNAL GENERATION ##
        self.res_losses = 100 * self.opt_loss * (self.res_X1.sum(axis=1) + self.res_X2.sum(axis=1)) / (self.res_PG.sum(axis=1) + self.res_WIND.sum(axis=1))

        ## CALCULATE KINETIC ENERGY ##

        self.res_ekin = pd.DataFrame(dtype=float,data=0,index=self.timerange,columns=['model','data'])

        for a in self.syncareas:
            for g in self.res_gen[a]:
                if g in ['Thermal','Hydro','Nuclear']: # add inertia contribution
                    self.res_ekin.loc[:,'model'] += self.res_gen[a].loc[:,g] * \
                                                       self.opt_inertia_constants[self.area_to_country[a]][g] / \
                                                       self.opt_inertia_pf[self.area_to_country[a]][g] / \
                                                       self.opt_inertia_cf[self.area_to_country[a]][g]
                    self.res_ekin.loc[:,'data'] += self.entsoe_data[a].loc[:,g] * MWtoGW * \
                                                      self.opt_inertia_constants[self.area_to_country[a]][g] / \
                                                      self.opt_inertia_pf[self.area_to_country[a]][g] / \
                                                      self.opt_inertia_cf[self.area_to_country[a]][g]
            # add contribution from run of river hydro
            if a in self.ror_areas:
                self.res_ekin.loc[:,'model'] += self.res_gen[a].loc[:,'HROR'] * \
                                                   self.opt_inertia_constants[self.area_to_country[a]]['Hydro'] / \
                                                   self.opt_inertia_pf[self.area_to_country[a]]['Hydro'] / \
                                                   self.opt_inertia_cf[self.area_to_country[a]]['Hydro']
            # add contribution from pumped hydro
            if a in self.pump_res_areas:
                self.res_ekin.loc[:,'model'] += self.res_gen[a].loc[:,'REL'] * \
                                                self.opt_inertia_constants[self.area_to_country[a]]['Hydro'] / \
                                                self.opt_inertia_pf[self.area_to_country[a]]['Hydro'] / \
                                                self.opt_inertia_cf[self.area_to_country[a]]['Hydro']

        #### CURTAILMENT STATISTICS ####
        curstat_cols = ['GWh','%','#','avg len','avg GW','avg %']

        ## WIND ##
        self.wcur_tot = curtailment_statistics(self.res_cur_WIND.sum(axis=1),self.up_WIND.sum(axis=1),curstat_cols)
        self.wcur_pa = pd.DataFrame(index=self.wind_areas,columns=curstat_cols)
        self.wcur_pc = pd.DataFrame(index=self.opt_countries,columns=curstat_cols)

        for a in self.wcur_pa.index:
            self.wcur_pa.loc[a,:] = curtailment_statistics(self.res_cur_WIND[a],self.up_WIND[a],curstat_cols)
        for c in self.wcur_pc.index:
            al = [a for a in self.country_to_areas[c] if a in self.wind_areas]
            self.wcur_pc.loc[c,:] = curtailment_statistics( \
                self.res_cur_WIND.loc[:,al].sum(axis=1), \
                self.up_WIND.loc[:,al].sum(axis=1), \
                curstat_cols)

        # SOLAR ##
        self.scur_tot = curtailment_statistics(self.res_cur_SOLAR.sum(axis=1),self.up_SOLAR.sum(axis=1),curstat_cols)
        self.scur_pa = pd.DataFrame(index=self.solar_areas,columns=curstat_cols)
        self.scur_pc = pd.DataFrame(index=self.opt_countries,columns=curstat_cols)

        for a in self.scur_pa.index:
            self.scur_pa.loc[a,:] = curtailment_statistics(self.res_cur_SOLAR[a],self.up_SOLAR[a],curstat_cols)
        for c in self.scur_pc.index:
            al = [a for a in self.country_to_areas[c] if a in self.solar_areas]
            self.scur_pc.loc[c,:] = curtailment_statistics(
                self.res_cur_SOLAR.loc[:,al].sum(axis=1),
                self.up_SOLAR.loc[:,al].sum(axis=1),
                curstat_cols)

        ## TOTAL ## # NOTE: SOLAR HAS HIGHER PRIORITY -> NO SOLAR CURTAILMENT

        stat_thrs = 1e-4
        ls = ['wcur_pa','wcur_pc','wcur_tot','scur_pa','scur_pc','scur_tot']

        for name in ls:
            entity = self.__getattribute__(name)
            if type(entity) is pd.core.frame.DataFrame:
                entity[entity < stat_thrs] = 0

        # Compute some extra curtailment statistics
        # share of curtailment per area
        self.res_cur_per_area = pd.Series(index=self.wind_areas)
        for a in self.res_cur_per_area.index:
            self.res_cur_per_area.at[a] = 100 * self.res_cur_WIND[a].sum() / self.wcur_tot.at['GWh']
    
        # share of curtailment per month
        self.res_cur_per_month = pd.Series(index=range(1,13))
        for month in self.res_cur_per_month.index:
            self.res_cur_per_month.at[month] = 100 * self.res_cur_WIND.loc[[d for d in self.timerange if d.month == month],:].sum().sum() / self.wcur_tot.at['GWh']
    
        # share of curtailment per hour
        self.res_cur_per_hour = pd.Series(index=range(24))
        for hour in self.res_cur_per_hour.index:
            self.res_cur_per_hour.at[hour] = 100 * self.res_cur_WIND.loc[[h for h in self.timerange if h.hour == hour]].sum().sum() / self.wcur_tot.at['GWh']



        ## CALCULATE PRICES ##

        # for each price area: calculate total demand + net exports
        # check which price is required to produce corresponding amount in this area

        # put watervalues into hydro cost data
        for g in self.gen_data.index:
            if self.gen_data.at[g,'gtype'] == 'Hydro':
                self.gen_data.at[g,'c2'] = 0
                # get watervalues from duals
                if hasattr(self,'dual_constr_FIX_RESERVOIR'):
                    self.gen_data.at[g,'c1'] = self.dual_constr_FIX_RESERVOIR.at[self.gen_data.at[g,'area']]/MWtoGW

        # make supply curve for each region
        # Currently not used. May be used for comparing model supply curves to Nordpool price curves
        self.supply_curves = {}
        for area in self.areas:
            self.supply_curves[area] = SupplyCurve(gens=self.gen_data.loc[[g for g in self.gen_data.index if self.gen_data.at[g,'area'] == area],:])
        self.supply_curves['Tot'] = SupplyCurve(gens=self.gen_data)

    def add_hourly_maf_inflow(self):
        prt = self.opt_print['setup']
        for idx,date in enumerate(self.daysrange_weather_year):
            for a in self.inflow_daily_maf.columns:
                aidx = self.hydrores.index(a)
                self.inflow_hourly.iloc[idx*24:(idx+1)*24,aidx] += self.inflow_daily_maf.at[date,a] / 24

    def copy_data_to_model_year(self,df):
        """ 
        Given a dataframe with data for the same timerange but a different year compared to the model, make a new
        dataframe with the correct timerange. Note that it may be necessary to remove data for a leap day or add data
        for a leap day, if the the dataframe is for a leap year and the model is not, or vice versa
        Only works for data with hourly time resolution
        """

        # copy solar data into current year, for those hours which exist in both weather and model timerange (may be different due to leap years)
        df2 = pd.DataFrame(dtype=float,index=self.timerange_p1,columns=df.columns)

        self.idx_tups = []
        for t in self.timerange_p1:
            try: # note: trying to create a datetime object for feb 29 in year which is not leap year will give error
                tt = datetime.datetime(t.year+self.weather_year_diff,t.month,t.day,t.hour)
                self.idx_tups.append((tt,t))
            except ValueError:
                pass
        #
        # self.idx_tups = [(wt,mt) for wt,mt in [(datetime.datetime(t.year+self.weather_year_diff,t.month,t.day,t.hour),t) for t in self.timerange_p1]
        #                  if wt in df.index]
        self.weather_idx = [t[0] for t in self.idx_tups]
        self.model_idx = [t[1] for t in self.idx_tups]
        df2.loc[self.model_idx,:] = np.array(df.loc[self.weather_idx,:])
        # fill possible gaps due to leap year in model data but not in weather data
        dfnan = df2.isna().sum(axis=1)
        leap_days = [d for d in self.daysrange if d.month == 2 and d.day == 29]
        for ld in leap_days:
            start_idx = list(self.timerange).index(datetime.datetime(ld.year,ld.month,ld.day))
            if dfnan.iat[start_idx] == df2.shape[1]: # all missing, fill nan values
                if start_idx > 23: # fill with previous day
                    df2.iloc[start_idx:start_idx+24,:] = np.array(df2.iloc[start_idx-24:start_idx,:])
                else: # fill with next day
                    df2.iloc[start_idx:start_idx+24,:] = np.array(df2.iloc[start_idx+24:start_idx+48,:])
        return df2

    def impute_capacity_values(self):
        prt = self.opt_print['setup']
        #%% impute exchange capacities with fixed values
        nnan = self.exchange_capacity.isna().sum()
        for c in self.exchange_capacity.columns:
            if nnan.at[c]:
                if nnan.at[c] > self.nPeriods // 4:
                    if prt:
                        print(f'Imputing {nnan.at[c]} missing capacity values for {c} with nominal values')
                a1 = c.split(self.area_sep_str)[0]
                a2 = c.split(self.area_sep_str)[1]
                # find row in exchange capacity table
                for i,s in nordpool_capacities.iterrows():
                    if s['from'] == a1 and s['to'] == a2:
                        cap = s['c1']
                        break
                    elif s['to'] == a1 and s['from'] == a2:
                        cap = s['c2']
                        break
                self.exchange_capacity.loc[self.exchange_capacity[c].isna(),c] = cap

    def setup_reservoir_values(self):
        prt = self.opt_print['setup']

        # get starting value and end value for reservoir content
        dates = [str_to_date(self.starttime),str_to_date(self.endtime)]
        self.reservoir_fix = pd.DataFrame(dtype=float,index=dates,columns=self.hydrores)
        reservoir_tmp = interp_time(dates,self.reservoir)
        if self.opt_reservoir_data_normalized:
            for a in self.reservoir_fix.columns:
                if a in reservoir_tmp.columns:
                    self.reservoir_fix.loc[dates,a] = reservoir_tmp.loc[dates,a]*self.reservoir_capacity[a]
        else:
            self.reservoir_fix.loc[dates,reservoir_tmp.columns] = GWtoTW*reservoir_tmp.loc[dates,reservoir_tmp.columns]

        for a in self.hydrores:
            if np.isnan(self.reservoir_fix.at[dates[0],a]):
                if prt:
                    print(f'Using default reservoir filling rates for {a}')
                self.reservoir_fix.at[dates[0],a] = self.opt_reservoir_start_fill*self.reservoir_capacity[a]
            if np.isnan(self.reservoir_fix.at[dates[1],a]):
                self.reservoir_fix.at[dates[1],a] = self.opt_reservoir_end_fill*self.reservoir_capacity[a]

        # interpolate reservoir values for initialization of variables
        if self.opt_hydro_daily:
            tidx = self.daysrange_p1
        else:
            tidx = self.timerange_p1
        self.reservoir_interp = interp_time(tidx,self.reservoir_fix)

    def interpolate_inflow(self):
        pass
        prt = self.opt_print['setup']

        ## INTERPOLATE RESERVOIR INFLOW
        if prt:
            print('Interpolate weekly reservoir inflow')

        self.inflow_hourly_tmp = interpolate_weekly_values(self.inflow,method=self.opt_inflow_interp)

        # copy data to simulated year
        self.inflow_hourly = pd.DataFrame(dtype=float,columns=self.hydrores,index=self.timerange)
        icols = [i for i,c in enumerate(self.hydrores) if c in self.inflow_hourly_tmp.columns]
        self.inflow_hourly.loc[:,self.inflow_hourly_tmp.columns] = \
            np.array(self.inflow_hourly_tmp.iloc[self.inflow_offset:self.inflow_offset+self.nPeriods,icols])
        self.inflow_hourly[self.inflow_hourly < 0] = 0 # replace negative values

        # adding constant inflow, can be done after interpolation
        for a in self.hydrores:
            if a not in self.inflow_hourly_tmp.columns:
                if a in self.opt_default_inflow_area:
                    self.inflow_hourly[a] = self.opt_default_inflow_area[a] / 168
                else:
                    self.inflow_hourly[a] = self.opt_default_inflow / 168

    def impute_values(self,data_series=['demand'],limit=20,prt=True):
        # prt = self.opt_print['setup']
        """ Replace missing values in data by linear interpolation """
        for name in data_series:
            if name in self.opt_impute_constant:
                constant = self.opt_impute_constant[name]
            else:
                constant = None
            entity = self.__getattribute__(name)
            entity.interpolate(method='linear',inplace=True,limit=limit)
            entity.fillna(method='bfill',limit=2,inplace=True)
            entity.fillna(method='ffill',limit=2,inplace=True)
            if constant is not None:
                # fill all remaining nans with constant
                nnan = entity.isna().sum().sum()
                if nnan > 0:
                    if prt:
                        print(f'Imputing {nnan} constant values in {name}')
                    entity.fillna(constant,inplace=True)
            else: # count missing values
                nnan = entity.isna().sum().sum()
                if nnan:
                    if prt:
                        print(f'Impute incomplete: {nnan} remaining missing values in {name}')


    def print_rmse(self):

        rmse_area = self.res_rmse_area.copy()
        rmse_area.loc['Avg',:] = rmse_area.mean()

        rmse_area_rel = self.res_rmse_area.copy()
        norm_cols = ['Prod','Hydro','Thermal','Nuclear']
        rmse_area_rel.loc[:,norm_cols] = rmse_area_rel.loc[:,norm_cols] / self.res_rmse_area_norm.loc[:,norm_cols]
        rmse_area_rel.loc['Avg',:] = rmse_area_rel.mean()
        
        rmse_conn = self.res_rmse_intcon.copy()
        rmse_conn_rel = self.res_rmse_intcon.copy()
        rmse_conn_ext = self.res_rmse_extcon.copy()
        rmse_conn_ext_rel = self.res_rmse_extcon.copy()

        rmse_conn_rel['RMSE'] = rmse_conn_rel['RMSE'] / self.res_rmse_intcon_norm
        rmse_conn_ext_rel['RMSE'] = rmse_conn_ext_rel['RMSE'] / self.res_rmse_extcon_norm

        rmse_conn.at['Avg', 'RMSE'] = np.mean(rmse_conn['RMSE'])
        rmse_conn_ext.at['Avg', 'RMSE'] = np.mean(rmse_conn_ext['RMSE'])
        rmse_conn_rel.at['Avg', 'RMSE'] = np.mean(rmse_conn_rel['RMSE'])
        rmse_conn_ext_rel.at['Avg', 'RMSE'] = np.mean(rmse_conn_ext_rel['RMSE'])
    
        #%%
        na_rep = '-'
    
        area_formatters = {
            'Hydro':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Thermal':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Nuclear':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Prod':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Price':lambda x:f'{x:0.1f}' if not np.isnan(x) else na_rep,
        }

        area_formatters_rel = {
            'Hydro':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Thermal':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Nuclear':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Prod':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
            'Price':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
        }
    
        conn_formatters = {
            'From':lambda x:x if type(x) is str else na_rep,
            'To':lambda x:x if type(x) is str else na_rep,
            'RMSE':lambda x:f'{x:0.3f}' if not np.isnan(x) else na_rep,
        }

        #%% print tables to text file
        with open(self.res_path/f'errors.txt','wt') as f:
            with redirect_stdout(f):
                print(self.name + '\n')
                print(f'-- AREA {self.opt_err_labl} --')
                print(rmse_area)
    
                print('\n')
                print(f'-- AREA RELATIVE {self.opt_err_labl} --')
                print(rmse_area_rel)
    
                print('\n')
                print(f'-- INTERNAL CONNECTION {self.opt_err_labl} --')
                print(rmse_conn)
    
                print('\n')
                print(f'-- INTERNAL RELATIVE CONNECTION {self.opt_err_labl} --')
                print(rmse_conn_rel)
    
                print('\n')
                print(f'-- EXTERNAL CONNECTION {self.opt_err_labl} --')
                print(rmse_conn_ext)
    
                print('\n')
                print(f'-- EXTERNAL RELATIVE CONNECTION {self.opt_err_labl} --')
                print(rmse_conn_ext_rel)
    
        #%% print tables to latex file
        with open(self.res_path/f'errors.tex','wt') as f:
            with redirect_stdout(f):
                print(self.name+'\n')
    
                print(f'-- AREA {self.opt_err_labl} --')
                rmse_area.to_latex(f,formatters=area_formatters,
                                   header=['Prod. [GWh]','Hydro [GWh]','Thermal [GWh]','Nuclear [GWh]','Price [EUR/MWh]'])
    
                print('\n')
                print(f'-- AREA RELATIVE {self.opt_err_labl} --')
                rmse_area_rel.to_latex(f,formatters=area_formatters_rel,
                                       header=['Prod.','Hydro','Thermal','Nuclear','Price [EUR]'])
    
                print('\n')
                print(f'-- INTERNAL CONNECTION {self.opt_err_labl} --')
                rmse_conn.to_latex(f,header=['From','To','RMSE [GWh]'],formatters=conn_formatters)
    
                print('\n')
                print(f'-- INTERNAL RELATIVE CONNECTION {self.opt_err_labl} --')
                rmse_conn_rel.to_latex(f,header=['From','To','RMSE'],formatters=conn_formatters)
    
                print('\n')
                print(f'-- EXTERNAL CONNECTION {self.opt_err_labl} --')
                rmse_conn_ext.to_latex(f,header=['From','To','RMSE [GWh]'],formatters=conn_formatters)
    
                print('\n')
                print(f'-- EXTERNAL RELATIVE CONNECTION {self.opt_err_labl} --')
                rmse_conn_ext_rel.to_latex(f,header=['From','To','RMSE'],formatters=conn_formatters)

    def print_hydro_table(self):
        # path = 'D:/NordicModel/InputData/'
        filename = 'hydro_table.txt'
        df = pd.DataFrame(dtype=float,columns=['hydro','res','respump','pump','ror'],index=self.hydrores)
    
        for a in self.hydrores:
    
            gidx = next(g for g in self.idx_gen if self.gen_data.at[g,'area'] == a and self.gen_data.at[g,'gtype'] == 'Hydro')
            df.at[a,'hydro'] = self.gen_data.at[gidx,'pmax']
            if a in self.opt_pump_reservoir:
                df.at[a,'respump'] = self.opt_pump_reservoir[a]
            if a in self.opt_reservoir_capacity:
                df.at[a,'res'] = self.opt_reservoir_capacity[a]
            if a in self.ror_areas:
                df.at[a,'ror'] = self.opt_ror_fraction[a]
            if a in self.opt_pump_capacity:
                df.at[a,'pump'] = self.opt_pump_capacity[a]
    
        na_rep = '-'
        headers = ['Hydro max (MWh)', 'Reservoir (GWh)','Pump reservoir (GWh)','Pump max (MWh)','ROR share']
        with open(Path(self.res_path) / filename,'wt') as f:
            df.to_latex(f,header=headers,
                        formatters={
                            'hydro':lambda x:str(np.round(x,1)) if not np.isnan(x) else na_rep,
                            'res':lambda x:str(np.round(x,1)) if not np.isnan(x) else na_rep,
                            'respump':lambda x:str(np.round(x,1)) if not np.isnan(x) else na_rep,
                            'pump':lambda x:str(np.round(x,1)) if not np.isnan(x) else na_rep,
                            'ror':lambda x:str(np.round(x,2)) if not np.isnan(x) else na_rep,
                        })

    def print_renewable_table(self,countries=['SE','DK','NO','FI']):
        #%% print wind and solar capacity tables, by area or country
        areas = [a for a in all_areas if area_to_country[a] in countries]
    
        if areas is None:
            pass
    
        wind_cap_area = pd.DataFrame(dtype=int,index=areas,columns=['onshore','offshore'])
        wind_cap_country = pd.DataFrame(0,dtype=int,
                                        index=countries,
                                        columns=['onshore','offshore','solar'])
    
        # pv_cap_area = pd.DataFrame(dtype=int,index=self.solar_areas,columns=['mw'])
        # pv_cap_country = pd.DataFrame(0,dtype=int,index=[c for c in self.opt_countries if \
        #                                                sum([1 for a in self.solar_areas if a in self.country_to_areas[c]])],columns=['mw'])
    
        for a in areas:
            val = self.opt_wind_capacity_onsh[a]
            wind_cap_area.at[a,'onshore'] = val
            wind_cap_country.at[self.area_to_country[a],'onshore'] += val
            val = self.opt_wind_capacity_offsh[a]
            wind_cap_area.at[a,'offshore'] = val
            wind_cap_country.at[self.area_to_country[a],'offshore'] += val
    
            sval = int(np.round(self.solar_capacity[a]*GWtoMW,-1))
            wind_cap_area.at[a,'solar'] = sval
            wind_cap_country.at[self.area_to_country[a],'solar'] += sval
    
        #%
        wind_cap_area.loc['Tot',:] = wind_cap_area.sum()
        wind_cap_country.loc['Tot',:] = wind_cap_country.sum()
        # for a in self.solar_areas:
        #     val = int(self.solar_capacity[a]*GWtoMW)
        #     pv_cap_area.at[a,'mw'] = val
        #     pv_cap_country.at[self.area_to_country[a],'mw'] += val
    
        wind_cap_area = wind_cap_area.astype(int)
        wind_cap_country = wind_cap_country.astype(int)
    
        with open(self.res_path/f'renewable_capacity.tex','wt') as f:
            with redirect_stdout(f):
                print('--- RENEWABLE CAPACITY BY AREA ---')
                wind_cap_area.to_latex(f)
                print('--- RENEWABLE CAPACITY BY COUNTRY ---')
                wind_cap_country.to_latex(f)
                # print('--- SOLAR CAPACITY BY AREA ---')
                # pv_cap_area.to_latex(f)
                # print('--- SOLAR CAPACITY BY COUNTRY ---')
                # pv_cap_country.to_latex(f)


    def get_fig(self):
        f = self.f
        # f = plt.gcf()
        f.clf()
        f.set_size_inches(6.4,4.8)
        f.set_tight_layout(False)
        return f

    def plot_figures(self):
        prt = self.opt_print['postprocess']
        self.plot_vals = EmptyObject()

        plt.ioff() # don't create new figures
        plt.rc('text', usetex=False)
        self.f = plt.figure()
        # for which categories to use inset
        self.plot_vals.plot_inset = {
            'Hydro':True,
            'Thermal':True,
            'Nuclear':False,
        }
        # settings without inset
        self.plot_vals.legend_pos = 'best'
        self.plot_vals.simple_plot_size = (17,10)
        # settings for inset
        self.plot_vals.myFmt = "%m-%d"

        if not hasattr(self,'fopt_inset_date'): # default options for old models
            self.fopt_inset_date = None
            self.fopt_inset_days = 5

        if self.fopt_inset_date is None:
            self.fopt_inset_date = self.starttime
        self.plot_vals.inset_idx = pd.date_range(start=str_to_date(self.fopt_inset_date),
                                                 end=min((str_to_date(self.fopt_inset_date) \
                                                          + datetime.timedelta(days=self.fopt_inset_days)),
                                                         str_to_date(self.endtime)+datetime.timedelta(hours=-1)),
                                                 freq='H')
        self.plot_vals.ax2_width = 0.55
        # legend position with inset
        self.plot_vals.bbox = (0.72,1.02) # horizontal, vertical
        self.plot_vals.inset_height_ratio = [1,1.6]

        self.plot_vals.colors = {'Hydro':'skyblue',
                  'Slow':'#ff7f0e',
                  'Fast':'#d62728',
                  'Nuclear':'#8c564b',
                  'Wind':'#2ca02c',
                  'Thermal':'#ff7f0e',
                  'Solar':'khaki',
                  'HROR':'darkorchid',
                  }

        if self.fopt_show_rmse:
            self.plot_vals.annstr = f'{self.opt_err_labl}: ' + '{1:.3}\n' + f'N{self.opt_err_labl}: ' + '{0:.3}'
        else:
            self.plot_vals.annstr = f'N{self.opt_err_labl}: ' + '{0:.3}'

        # list of all possible plots, make sure all items are present in fopt_plots, set to False for missing values
        all_plots = ['gentype','gentot','gentot_bar','renewables','transfer_internal','transfer_external',
                     'reservoir','price','losses','load_curtailment','inertia','hydro_duration','wind_curtailment'
        ]
        for f in all_plots:
            if f not in self.fopt_plots:
                self.fopt_plots[f] = False


        self.plot_gentype()
        self.plot_gentot()
        if self.fopt_plots['renewables'] and not self.fopt_no_plots:
            self.plot_renewables()
        self.plot_transfer()
        self.plot_reservoir()
        self.plot_price()
        if self.fopt_plots['wind_curtailment'] and not self.fopt_no_plots:
            self.plot_wind_curtailment()
        self.plot_miscellaneous()
        plt.ion()

    def plot_gentype(self):
        prt = self.opt_print['postprocess']
        dummy = True
        for area in self.areas:
            for gtype in self.generators_def[area]:
                if gtype == 'Hydro':
                    hydro_prod = pd.Series(0.0,index=self.timerange)
                    for var in ['Hydro','HROR','REL']:
                        if var in self.res_gen[area]:
                            hydro_prod += self.res_gen[area][var]
                    irmse,irmse_rel,norm=err_func(self.entsoe_data[area][gtype]*MWtoGW,hydro_prod)
                else:
                    pass
                    irmse,irmse_rel,norm=err_func(self.entsoe_data[area][gtype]*MWtoGW,self.res_gen[area][gtype])

                self.res_rmse_area.at[area,gtype] = irmse
                self.res_rmse_area_norm.at[area,gtype] = norm

                if self.fopt_plots['gentype'] and not self.fopt_no_plots:
                    if prt and dummy:
                        dummy = False
                        print('Plot gentype')
                    ############ collect plot data ###########
                    plot_data = pd.DataFrame(index=self.timerange,columns=['res','ror','model','data','pump','rel','up','lo'])

                    if gtype == 'Hydro':
                        plot_data['res'] = self.res_gen[area]['Hydro']
                        if area in self.ror_areas:
                            plot_data['ror'] = self.res_gen[area]['HROR']
                        if area in self.pump_areas:
                            plot_data['pump'] = -self.res_PUMP[area]
                        if area in self.pump_res_areas:
                            plot_data['rel'] = self.res_gen[area]['REL']
                        plot_data['model'] = hydro_prod
                    else:
                        plot_data['model'] = self.res_gen[area][gtype]
                    plot_data['data'] = self.entsoe_data[area][gtype]*MWtoGW

                    if gtype == 'Hydro' and area in self.ror_areas:
                        hgens = [g for g in self.gen_data.index if self.gen_data.at[g,'area'] == area \
                                 and self.gen_data.at[g,'gtype']=='Hydro']
                        plot_data['up'] = self.gen_data.loc[hgens,'pmax'].sum() * MWtoGW
                        plot_data['lo'] = self.gen_data.loc[hgens,'pmin'].sum() * MWtoGW

                    else:
                        plot_data['up'] = [sum([self.max_PG[(i,t)] for i in self.idx_gen
                                                if self.gen_data.at[i,'area'] == area and self.gen_data.at[i,'gtype'] == gtype])
                                           for t in self.idx_time]
                        plot_data['lo'] = [sum([self.min_PG[(i,t)] for i in self.idx_gen
                                                if self.gen_data.at[i,'area'] == area and self.gen_data.at[i,'gtype'] == gtype])
                                           for t in self.idx_time]

                    ########### plot figure ############
                    if self.plot_vals.plot_inset[gtype]: # figure with inset

                        # get axes
                        # f = plt.gcf()
                        # f.clf()
                        f = self.get_fig()
                        # f.set_size_inches()
                        f.set_size_inches(6.8,4.8)
                        ax2,ax1 = f.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                        # f,(ax2,ax1) = plt.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                        pos = ax2.get_position()
                        ax2.set_position([pos.x0,pos.y0,self.plot_vals.ax2_width,pos.height])

                        # main plot
                        if gtype == 'Hydro':
                            if area in self.pump_areas and area in self.ror_areas:
                                if area in self.pump_res_areas:
                                    ptypes = ['res','ror','rel','model','data','pump','up','lo']
                                    pcolors = [colors['Hydro'],colors['HROR'],colors['REL'],'C0','C1',colors['PUMP'],'k','k']
                                    pstyles = ['--',':','--','-','-','-','--','--']
                                    plegends = ['reservoir','run of river','pump release','model','data','pumping']
                                else:
                                    ptypes = ['res','ror','model','data','pump','up','lo']
                                    pcolors = [colors['Hydro'],colors['HROR'],'C0','C1',colors['PUMP'],'k','k']
                                    pstyles = ['--',':','-','-','-','--','--']
                                    plegends = ['reservoir','run of river','model','data','pumping']
                            elif area in self.pump_areas: # pumping but no run of river
                                if area in self.pump_res_areas:
                                    ptypes = ['res','rel','model','data','pump','up','lo']
                                    pcolors = [colors['Hydro'],colors['REL'],'C0','C1',colors['PUMP'],'k','k']
                                    pstyles = ['--','--','-','-','-','--','--']
                                    plegends = ['reservoir','pump release','model','data','pumping']
                                else: # pumping, no run of river
                                    ptypes = ['model','data','pump','up','lo']
                                    pcolors = ['C0','C1',colors['PUMP'],'k','k']
                                    pstyles = ['-','-','-','--','--']
                                    plegends = ['model','data','pumping']
                            elif area in self.ror_areas: # only ror
                                ptypes = ['res','ror','model','data','up','lo']
                                pcolors = [colors['Hydro'],colors['HROR'],'C0','C1','k','k']
                                pstyles = ['--',':','-','-','--','--']
                                plegends = ['reservoir','run of river','model','data']
                            else: # standard plot, neither pumping or run of river
                                ptypes = ['model','data','up','lo']
                                pcolors = ['C0','C1','k','k']
                                pstyles = ['-','-','--','--']
                                plegends = ['model','data']

                            plot_data.loc[:,ptypes].plot(ax=ax1,color=pcolors,style=pstyles)
                            ax1.legend(plegends,title=self.plot_vals.annstr.format(irmse_rel,irmse),
                                       bbox_to_anchor=self.plot_vals.bbox)

                        else:
                            plot_data.loc[:,['model','data','up','lo']].plot(ax=ax1,color=['C0','C1','k','k'],
                                                                             style=['-','-','--','--'])
                            ax1.legend(['model','data',],title=self.plot_vals.annstr.format(irmse_rel,irmse),
                                       bbox_to_anchor=self.plot_vals.bbox)

                        ax1.set_ylabel('GWh')
                        if self.fopt_use_titles:
                            f.suptitle('{2}: {1} production {0}'.format(area,gtype,self.name))
                        ax1.set_zorder(1)
                        ax1.grid()

                        # remove line breaks from ticks
                        compact_xaxis_ticks(f,ax1)

                        # inset
                        ax2.plot(plot_data.loc[self.plot_vals.inset_idx,['model','data']])
                        ax2.grid()
                        ax2.xaxis.set_major_formatter(mdates.DateFormatter(self.plot_vals.myFmt))
                        ax2.xaxis.set_major_locator(mdates.DayLocator())


                    else: # only one figure
                        f = self.get_fig()
                        ax = f.subplots()
                        if gtype == 'Hydro':
                            plot_data.plot(ax=ax,color=[colors['Hydro'],colors['HROR'],'C0','C1','k','k'],style=['--',':','-','-','--','--'])
                            plt.legend(['reservoir','run of river','model','data',],title=self.plot_vals.annstr.format(irmse_rel,irmse),loc=self.plot_vals.legend_pos)
                        else:
                            plot_data.loc[:,['model','data','up','lo']].plot(ax=ax,color=['C0','C1','k','k'],style=['-','-','--','--'])
                            plt.legend(['model','data',],title=self.plot_vals.annstr.format(irmse_rel,irmse),loc=self.plot_vals.legend_pos)

                        plt.grid()
                        if self.fopt_use_titles:
                            plt.title('{2}: {1} production {0}'.format(area,gtype,self.name))
                        plt.ylabel('GWh')

                        plt.tight_layout()
                        plt.gcf().set_size_inches(self.plot_vals.simple_plot_size[0]/cm_per_inch,self.plot_vals.simple_plot_size[1]/cm_per_inch)

                    ############ save figure ##########

                    plt.savefig(self.fig_path / 'gen_by_type_{0}_{1}.png'.format(area,gtype))
                    if self.fopt_eps:
                        plt.savefig(self.fig_path / 'gen_by_type_{0}_{1}.eps'.format(area,gtype),
                                    dpi=self.fopt_dpi_qual)
                    if not self.plot_vals.plot_inset[gtype]:
                        for w in self.fopt_plot_weeks:
                            t1,t2 = week_to_range(w,int(self.starttime[:4]))
                            plt.xlim([t1,t2])
                            plt.savefig(self.fig_path /  'gen_by_type_{0}_{1}_w{2}.png'.format(area,gtype,w))
                            if self.fopt_eps:
                                plt.savefig(self.fig_path /  'gen_by_type_{0}_{1}_w{2}.eps'.format(area,gtype,w),
                                            dpi=self.fopt_dpi_qual)

                    # plt.clf()
                    

    def plot_gentot(self):
        prt = self.opt_print['postprocess']
        dummy = True
        ## PLOT GENERATION PER AREA ##
        for area in self.areas:
            irmse,irmse_rel,norm = err_func(self.entsoe_data[area]['Tot']*MWtoGW,self.res_gen[area].sum(axis=1))

            self.res_rmse_area.at[area,'Prod'] = irmse
            self.res_rmse_area_norm.at[area,'Prod'] = norm


            if self.fopt_plots['gentot_bar'] and not self.fopt_no_plots:
                if prt and dummy:
                    dummy = False
                    print('Plot gentot stacked')
                # print(f'Bar {area}')
                ############ STACKED GENERATION OF DIFFERENT TYPES ############
                # ax1 = f.add_axes()
                f = self.get_fig()
                ax1 = f.subplots()
                self.res_gen[area].plot.area(ax=ax1,color = [colors[f] for f in self.res_gen[area].columns])

                self.res_D[area].plot(ax=ax1,color='black',label='Demand')
                self.res_exp[area].sum(axis=1).plot(ax=ax1,color='black',linestyle='--',label='Exports')
                if area in self.pump_areas:
                    self.res_PUMP[area].plot(ax=ax1,color=colors['PUMP'],label='Pumping')

                plt.legend(ax1.legendlabels)
                if self.fopt_use_titles:
                    plt.title('{0}: Production {1}'.format(self.name,area))
                plt.grid()
                plt.ylabel('GWh')
                ylim_padding = 0.5
                ax1.set_ylim([min([min(l.get_ydata()) for l in ax1.lines])-ylim_padding,max([max(l.get_ydata()) for l in ax1.lines])+ylim_padding])

                plt.savefig(self.fig_path/'gen_area_plot_{0}.png'.format(area))
                if self.fopt_eps:
                    plt.savefig(self.fig_path/'gen_area_plot_{0}.eps'.format(area),dpi=self.fopt_dpi_qual)
                for w in self.fopt_plot_weeks:
                    # find daterange for this week
                    t1,t2 = week_to_range(w,int(self.starttime[:4]))
                    plt.xlim([t1,t2])
                    plt.savefig(self.fig_path/'gen_area_plot_{0}_w{1}.png'.format(area,w))
                    if self.fopt_eps:
                        plt.savefig(self.fig_path/'gen_area_plot_{0}_w{1}.eps'.format(area,w))

        dummy = True
        for area in self.areas:
            # f = self.get_fig()

            ######## TOTAL GENERATION, COMPARE NORDPOOL ##############
            if self.fopt_plots['gentot'] and not self.fopt_no_plots:
                if prt and dummy:
                    dummy = False
                    print('Plot total generation')
                plot_data = pd.DataFrame(index=self.timerange,columns=['model','data'])
                plot_data['model'] = self.res_gen[area].sum(axis=1)
                plot_data['data'] = self.entsoe_data[area]['Tot']*MWtoGW

                # get axes
                f = self.get_fig()
                # f = plt.gcf()
                # f.clf()
                # print(f'Gentot {area}')
                ax2,ax1 = f.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                # f,(ax2,ax1) = plt.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                pos = ax2.get_position()
                ax2.set_position([pos.x0,pos.y0,self.plot_vals.ax2_width,pos.height])

                rmse = self.res_rmse_area.at[area,'Prod']
                norm = self.res_rmse_area_norm.at[area,'Prod']
                rmse_rel = rmse/norm
                # main plot
                plot_data.plot(ax=ax1)
                ax1.legend(['model','data',],title=self.plot_vals.annstr.format(rmse_rel,rmse),
                           bbox_to_anchor=self.plot_vals.bbox)
                ax1.set_ylabel('GWh')
                if self.fopt_use_titles:
                    f.suptitle('{0}: Total production {1}'.format(self.name,area))
                ax1.set_zorder(1)
                ax1.grid()
                # remove line breaks from ticks
                compact_xaxis_ticks(f,ax1)

                # inset
                ax2.plot(plot_data.loc[self.plot_vals.inset_idx,:])
                ax2.grid()
                ax2.xaxis.set_major_formatter(mdates.DateFormatter(self.plot_vals.myFmt))
                ax2.xaxis.set_major_locator(mdates.DayLocator())

                plt.savefig(self.fig_path/f'gentot_{area}.png')
                if self.fopt_eps:
                    plt.savefig(self.fig_path/f'gentot_{area}.eps',dpi=self.fopt_dpi_qual)



        ## PLOT GENERATION FOR COUNTRIES ##
        if self.fopt_plots['gentot_bar'] and not self.fopt_no_plots:
            if prt:
                print('Plot gentot for countries')

            for c in self.multi_area_countries:
                # print(f'Bar {c}')
                self.get_fig()
                ax = f.subplots()
                self.res_gen[c].plot.area(ax=ax,color = [colors[f] for f in self.res_gen[c].columns])
                plt.plot(ax.lines[0].get_xdata(),self.res_D.loc[:,[col for col in self.demand.columns if c in col]].sum(axis=1),'k')
                plt.plot(ax.lines[0].get_xdata(),self.res_exp[c].sum(axis=1),'--k')
                plt.legend(ax.legendlabels + ['Demand','Export'])
                ylim_padding = 0.5
                ax.set_ylim([min([min(l.get_ydata()) for l in ax.lines])-ylim_padding,max([max(l.get_ydata()) for l in ax.lines])+ylim_padding])
                plt.grid()
                plt.ylabel('GWh/h')
                if self.fopt_use_titles:
                    plt.title('{0}: Production {1}'.format(self.name,c))
                plt.savefig(self.fig_path/'gen_area_plot_{0}.png'.format(c))
                if self.fopt_eps:
                    plt.savefig(self.fig_path/'gen_area_plot_{0}.eps'.format(c),dpi=self.fopt_dpi_qual)
                for w in self.fopt_plot_weeks:
                    t1,t2 = week_to_range(w,int(self.starttime[:4]))
                    plt.xlim([t1,t2])
                    plt.savefig(self.fig_path/'gen_area_plot_{0}_w{1}.png'.format(c,w))
                    if self.fopt_eps:
                        plt.savefig(self.fig_path/'gen_area_plot_{0}_w{1}.eps'.format(c,w),dpi=self.fopt_dpi_qual)
                # using the bar plot creates a new figure
                
                # plt.cla()

    def plot_renewables(self):
        prt = self.opt_print['postprocess']
        if prt:
            print('Plot renewables')
        # plot wind and solar production
        wcolor = 'royalblue'
        scolor = 'gold'

        for area in [a for a in self.areas if a in self.wind_areas or a in self.solar_areas]:

            # get axes
            f = self.get_fig()
            ax2,ax1 = f.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
            pos = ax2.get_position()
            ax2.set_position([pos.x0,pos.y0,self.plot_vals.ax2_width,pos.height])

            if area in self.wind_areas:
                self.res_gen[area]['Wind'].plot(ax=ax1,label='Wind',color=wcolor)
            if area in self.solar_areas:
                self.res_gen[area]['Solar'].plot(ax=ax1,label='Solar',color=scolor)

            ax1.legend(bbox_to_anchor=self.plot_vals.bbox)
            ax1.set_ylabel('GWh')
            if self.fopt_use_titles:
                f.suptitle(f'{self.name}: Renewable generation {area}')
            ax1.set_zorder(1)
            ax1.grid()
            # remove line breaks from ticks
            compact_xaxis_ticks(f,ax1)

            # inset
            if area in self.wind_areas:
                ax2.plot(self.res_gen[area]['Wind'].loc[self.plot_vals.inset_idx],color=wcolor)
            if area in self.solar_areas:
                ax2.plot(self.res_gen[area]['Solar'].loc[self.plot_vals.inset_idx],color=scolor)
            ax2.grid()
            ax2.xaxis.set_major_formatter(mdates.DateFormatter(self.plot_vals.myFmt))
            ax2.xaxis.set_major_locator(mdates.DayLocator())

            plt.savefig(self.fig_path/f'renewables_{area}.png')
            if self.fopt_eps:
                plt.savefig(self.fig_path/f'renewables_{area}.eps',dpi=self.fopt_dpi_qual)

    def plot_transfer(self):
        prt = self.opt_print['postprocess']
        dummy = True
        # get data

        for conn in self.xtrans_int.index:

            if self.fopt_plots['transfer_internal'] or self.fopt_calc_rmse['transfer']:
                cname = self.xtrans_int.at[conn,'label_fw']
                if cname in self.df_exchange_rmse.columns:
                    irmse,irmse_rel,norm = err_func(self.df_exchange_rmse[cname]*MWtoGW,self.res_xint[conn])

                else:
                    irmse = np.nan
                    irmse_rel = np.nan
                    norm = np.nan

                self.res_rmse_intcon.at[conn,'RMSE'] = irmse
                self.res_rmse_intcon_norm.at[conn] = norm

            if self.fopt_plots['transfer_internal'] and not self.fopt_no_plots:
                if prt and dummy:
                    dummy = False
                    print('Plot transfer')

                plot_data = pd.DataFrame(index=self.timerange,columns=['model','data','up','lo'])
                plot_data['model'] = self.res_X1[conn]-self.res_X2[conn]
                # if not self.xint_to_exchange[conn] is None:
                if cname in self.df_exchange_rmse.columns:
                    plot_data['data'] = self.df_exchange_rmse.loc[:,cname]*MWtoGW
                # plot_data['up'] = self.up_X1.loc[:,conn]
                plot_data['up'] = [self.max_X1[(conn,t)] for t in self.idx_time]
                # plot_data['lo'] = -self.up_X2.loc[:,conn]
                plot_data['lo'] = [-self.max_X2[(conn,t)] for t in self.idx_time]
                # get axes
                # f = plt.gcf()
                # f.clf()
                f = self.get_fig()
                ax2,ax1 = f.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                # f,(ax2,ax1) = plt.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                pos = ax2.get_position()
                ax2.set_position([pos.x0,pos.y0,self.plot_vals.ax2_width,pos.height])

                # main plot
                plot_data.plot(ax=ax1,color=['C0','C1','k','k'],style=['-','-','--','--'])
                if cname in self.df_exchange_rmse.columns:
                    labels = ['model','data']
                else:
                    labels = ['model']
                ax1.legend(labels,title=self.plot_vals.annstr.format(irmse_rel,irmse),bbox_to_anchor=self.plot_vals.bbox)
                ax1.set_ylabel('GWh')
                if self.fopt_use_titles:
                    f.suptitle('{2}: Transfer {0} -> {1}'.format(self.xtrans_int.at[conn,'from'],self.xtrans_int.at[conn,'to'],self.name))
                ax1.set_zorder(1)
                ax1.grid()
                # remove line breaks from ticks
                compact_xaxis_ticks(f,ax1)

                # inset
                ax2.plot(plot_data.loc[self.plot_vals.inset_idx,['model','data']])
                ax2.grid()
                ax2.xaxis.set_major_formatter(mdates.DateFormatter(self.plot_vals.myFmt))
                ax2.xaxis.set_major_locator(mdates.DayLocator())

                plt.savefig(self.fig_path/'xtrans_{0}-{1}.png'.format(self.xtrans_int.at[conn,'from'],self.xtrans_int.at[conn,'to']))
                if self.fopt_eps:
                    plt.savefig(self.fig_path/'xtrans_{0}-{1}.eps'.format(self.xtrans_int.at[conn,'from'],self.xtrans_int.at[conn,'to']),dpi=self.fopt_dpi_qual)

        # external variable connections
        for conn in self.fixed_price_connections:

            if self.fopt_plots['transfer_external'] or self.fopt_calc_rmse['transfer']:
                cname = self.xtrans_ext.at[conn,'label_fw']

                irmse,irmse_rel,norm=err_func(self.df_exchange_rmse[cname]*MWtoGW,self.res_XEXT[conn])

                self.res_rmse_extcon.at[conn,'RMSE'] = irmse
                self.res_rmse_extcon_norm.at[conn] = norm

            if self.fopt_plots['transfer_external'] and not self.fopt_no_plots:
                f = self.get_fig()
                ax = self.res_XEXT[conn].plot()
                plt.grid()
                plt.ylabel('GWh')
                # plot nordpool values
                # if not self.xext_to_exchange[conn] is None:
                plt.plot(ax.lines[0].get_xdata(),self.df_exchange_rmse.loc[:,self.xtrans_ext.at[conn,'label_fw']]*MWtoGW)
                # plot limits
                #                plt.plot([ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[-1]],[self.xtrans_ext.at[conn,'c1']*MWtoGW,self.xtrans_ext.at[conn,'c1']*MWtoGW],color='black',linestyle='--')
                #                plt.plot([ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[-1]],[-self.xtrans_ext.at[conn,'c2']*MWtoGW,-self.xtrans_ext.at[conn,'c2']*MWtoGW],color='black',linestyle='--')
                # plt.plot(ax.lines[0].get_xdata(),self.lo_XEXT[conn],color='black',linestyle='--')
                # plt.plot(ax.lines[0].get_xdata(),self.up_XEXT[conn],color='black',linestyle='--')
                plt.plot(ax.lines[0].get_xdata(),[self.min_XEXT[conn,t] for t in self.idx_time],color='black',linestyle='--')
                plt.plot(ax.lines[0].get_xdata(),[self.max_XEXT[conn,t] for t in self.idx_time],color='black',linestyle='--')


                # if not self.xext_to_exchange[conn] is None:
                plt.legend(['Model','Nordpool data','Max','Min'],title=self.plot_vals.annstr.format(irmse_rel,irmse),loc=self.plot_vals.legend_pos)
                # else:
                #     plt.legend(['Model','Max','Min'],loc=self.plot_vals.legend_pos)
                if self.fopt_use_titles:
                    plt.title('{2}: {0} -> {1}'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to'],self.name))
                plt.savefig(self.fig_path/'xtrans_{0}-{1}.png'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to']))
                if self.fopt_eps:
                    plt.savefig(self.fig_path/'xtrans_{0}-{1}.eps'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to']),dpi=self.fopt_dpi_qual)
                for w in self.fopt_plot_weeks:
                    t1,t2 = week_to_range(w,int(self.starttime[:4]))
                    plt.xlim([t1,t2])
                    plt.savefig(self.fig_path/'xtrans_{0}-{1}_w{2}.png'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to'],w))
                    if self.fopt_eps:
                        plt.savefig(self.fig_path/'xtrans_{0}-{1}_w{2}.eps'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to'],w),dpi=self.fopt_dpi_qual)

        # external fixed connections
        for conn in self.fixed_transfer_connections:

            if self.fopt_plots['transfer_external'] and not self.fopt_no_plots:
                f = self.get_fig()
                ax = self.res_xext[conn].plot()
                if self.fopt_use_titles:
                    plt.title('{2}: {0} -> {1} (fixed)'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to'],self.name))
                plt.grid()
                plt.ylabel('GWh')
                compact_xaxis_ticks(plt.gcf(),ax)


                plt.savefig(self.fig_path/'xtrans_{0}_{1}_fixed.png'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to']))
                if self.fopt_eps:
                    plt.savefig(self.fig_path/'xtrans_{0}_{1}_fixed.eps'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to']),dpi=self.fopt_dpi_qual)
                for w in self.fopt_plot_weeks:
                    t1,t2 = week_to_range(w,int(self.starttime[:4]))
                    plt.xlim([t1,t2])
                    plt.savefig(self.fig_path/'xtrans_{0}_{1}_fixed_w{2}.png'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to'],w))
                    if self.fopt_eps:
                        plt.savefig(self.fig_path/'xtrans_{0}_{1}_fixed_w{2}.eps'.format(self.xtrans_ext.at[conn,'from'],self.xtrans_ext.at[conn,'to'],w),dpi=self.fopt_dpi_qual)

                # plt.clf()
                

    def plot_reservoir(self):
        prt = self.opt_print['postprocess']
        from help_functions import interp_time
        if self.fopt_plots['reservoir'] and not self.fopt_no_plots:
            if prt:
                print('Plot reservoirs')
            if self.opt_reservoir_data_normalized:
                tmp = interp_time(dates=self.timerange_p1,df=self.reservoir)
                self.reservoir_hourly = pd.DataFrame(dtype=float,index=self.timerange_p1,columns=self.reservoir.columns)
                for a in self.reservoir_hourly.columns:
                    self.reservoir_hourly[a] = tmp[a] * self.reservoir_capacity[a]
            else:
                self.reservoir_hourly = GWtoTW * interp_time(dates=self.timerange_p1,df=self.reservoir)

            ## PLOT RESERVOIR CONTENT FOR EACH AREA ##
            for area in self.hydrores:

                if self.opt_hydro_daily:
                    plot_RES = pd.Series(index=self.daysrange_p1,dtype=float)
                    plot_RES.iat[0] = self.reservoir_fix.at[self.daysrange_p1[0],area]
                    plot_RES.iloc[1:] = np.array(self.res_RES[area])
                    tidx = self.daysrange_p1
                else:
                    plot_RES = self.res_RES[area]
                    tidx = self.timerange
                # ax = self.res_RES[area].plot()
                f = self.get_fig()
                ax = plot_RES.plot()
                xdata = ax.lines[0].get_xdata()
                # plt.plot(ax.lines[0].get_xdata(),self.reservoir_hourly.loc[self.res_RES[area].index,area])
                if area in self.reservoir_hourly:
                    plt.plot(xdata,self.reservoir_hourly.loc[tidx,area])
                # plot maximum reservoir value
                plt.plot([xdata[0],xdata[-1]],
                         [self.reservoir_capacity[area],self.reservoir_capacity[area]],'k--')
                ax.set_ylim([0,ax.get_ylim()[1]])
                # ylim_padding = 100
                # ax.set_ylim([min([min(l.get_ydata()) for l in ax.lines])-ylim_padding,
                #              max([max(l.get_ydata()) for l in ax.lines])+ylim_padding])
                plt.grid()
                plt.ylabel('TWh')

                plt.legend(['Model','Nordpool interpolated'])
                if self.fopt_use_titles:
                    plt.title('{0}: Hydro reservoir {1}'.format(self.name,area))
                compact_xaxis_ticks(plt.gcf(),ax)

                plt.savefig(self.fig_path/'hydrores_{0}.png'.format(area))
                if self.fopt_eps:
                    plt.savefig(self.fig_path/'hydrores_{0}.eps'.format(area),dpi=self.fopt_dpi_qual)

                # plot pump reservoir
                if area in self.pump_res_areas:
                    f = self.get_fig()
                    # ax.cla()
                    ax = self.res_PRES[area].plot(label='Model')
                    if area in self.opt_pump_reservoir:
                        xdata = ax.lines[0].get_xdata()
                        # plot maximum reservoir value
                        plt.plot([xdata[0],xdata[-1]],
                             [self.pump_reservoir[area],self.pump_reservoir[area]],'k--')

                    plt.ylabel('GWh')
                    plt.legend()
                    if self.fopt_use_titles:
                        plt.title(f'{self.name}: Pumping reservoir {area}')
                    plt.savefig(self.fig_path/f'pumpres_{area}.png')
                    if self.fopt_eps:
                        plt.savefig(self.fig_path/f'pumpres_{area}.eps')


        if self.fopt_plots['reservoir'] and not self.fopt_no_plots:
            ## PLOT TOTAL RESERVOIR CONTENT ##
            f = self.get_fig()
            cols = [h for h in self.hydrores if h in self.reservoir_hourly.columns]
            ax = self.res_RES.loc[:,cols].sum(axis=1).plot()
            plt.plot(ax.lines[0].get_xdata(),self.reservoir_hourly.loc[self.res_RES.index,cols].sum(axis=1))
            plt.grid()
            plt.ylabel('GWh')
            plt.legend(['Model','Nordpool interpolated'])
            if self.fopt_use_titles:
                plt.title('{0}: Total reservoir content'.format(self.name))
            compact_xaxis_ticks(plt.gcf(),ax)
            plt.savefig(self.fig_path/'total_reservoir.png')
            if self.fopt_eps:
                plt.savefig(self.fig_path/'total_reservoir.eps',dpi=self.fopt_dpi_qual)

            # plt.clf()
            

    def plot_price(self):
        prt = self.opt_print['postprocess']
        dummy = True
        ## PLOT PRICES FOR EACH AREA ##
        for area in self.areas:

            if self.fopt_calc_rmse['price'] or self.fopt_plots['price']:

                # use dual variables for prices
                irmse,irmse_rel,norm = err_func(self.df_price_internal[area],self.dual_constr_POWER_BALANCE[area]*GWtoMW)
                self.res_rmse_area.at[area,'Price'] = irmse
                self.res_rmse_area_norm.at[area,'Price'] = norm

            if self.fopt_plots['price'] and not self.fopt_no_plots:
                if prt and dummy:
                    dummy = False
                    print('Plot prices')
                # get axes
                # f = plt.gcf()
                # f.clf()
                f = self.get_fig()
                ax2,ax1 = f.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                # f,(ax2,ax1) = plt.subplots(2,1,gridspec_kw={'height_ratios':self.plot_vals.inset_height_ratio})
                pos = ax2.get_position()
                ax2.set_position([pos.x0,pos.y0,self.plot_vals.ax2_width,pos.height])

                (self.dual_constr_POWER_BALANCE[area]/MWtoGW).plot(ax=ax1)

                # plot calculated prices
                #plt.plot(ax.lines[0].get_xdata(),self.res_prices.loc[:,area])
                plt.plot(ax1.lines[0].get_xdata(),self.df_price_internal.loc[:,area])
                ylim_padding = 5
                ax1.set_ylim([min([min(l.get_ydata()) for l in ax1.lines])-ylim_padding,max([max(l.get_ydata())
                                                                                   for l in ax1.lines])+ylim_padding])

                ax1.set_zorder(1)
                ax1.grid()
                # remove line breaks from ticks
                compact_xaxis_ticks(f,ax1)

                # inset
                ax2.plot(self.dual_constr_POWER_BALANCE[area].loc[self.plot_vals.inset_idx]/MWtoGW)
                ax2.plot(self.df_price_internal.loc[self.plot_vals.inset_idx,area])
                ax2.grid()
                ax2.xaxis.set_major_formatter(mdates.DateFormatter(self.plot_vals.myFmt))
                ax2.xaxis.set_major_locator(mdates.DayLocator())

                ax1.set_ylabel('EUR/MWh')
                if self.fopt_use_titles:
                    f.suptitle('{1}: Price {0}'.format(area,self.name))
                ax1.legend(['Model dual price','Nordpool price',],title=self.plot_vals.annstr.format(irmse_rel,irmse),
                                          bbox_to_anchor=self.plot_vals.bbox)

                plt.savefig(self.fig_path/'price_{0}.png'.format(area))
                if self.fopt_eps:
                    plt.savefig(self.fig_path/'price_{0}.eps'.format(area),dpi=self.fopt_dpi_qual)
                for w in self.fopt_plot_weeks:
                    t1,t2 = week_to_range(w,int(self.starttime[:4]))
                    plt.xlim([t1,t2])
                    plt.savefig(self.fig_path/'price_{0}_w{1}.png'.format(area,w))
                    if self.fopt_eps:
                        plt.savefig(self.fig_path/'price_{0}_w{1}.eps'.format(area,w),dpi=self.fopt_dpi_qual)

                

        if self.fopt_plots['price'] and not self.fopt_no_plots:

            #%% plot average price for all areas ##
            price_df = pd.DataFrame(dtype=float,index=self.areas,columns=['model','data'])
            price_df['model'] = self.dual_constr_POWER_BALANCE.mean()*GWtoMW
            price_df['data'] = self.df_price_internal.mean()
    
            #%
            # f,ax = plt.subplots()
            f = self.get_fig()
            f.set_size_inches(6,3.8)
            # f.set_size_inches(6.4,4.8)
            ax = f.subplots()
            price_df.plot.bar(ax=ax)
            plt.grid()
            plt.ylabel('EUR/MWh')
            plt.ylim([10*np.floor_divide(price_df.min().min(),10),
                      10*(np.floor_divide(price_df.max().max(),10)+1)])
            plt.savefig(self.fig_path/'avg_prices.png')
            if self.fopt_eps:
                plt.savefig(self.fig_path/'avg_prices.eps')

    def plot_wind_curtailment(self):
        if self.opt_print['postprocess']:
            print('Plot wind curtailment')
        ## PLOT DURATION CURVES FOR WIND CURTAILMENT ##
        cur_thrsh = 1e-2
        xmaxs = []
        clinestyles = ['-','--',':','-.']
        clinewidths = [1.9,1.1]

        careas = [a for a in self.res_cur_WIND.columns if self.res_cur_WIND[a].sum()>cur_thrsh]
        cur = duration_curve(self.res_cur_WIND.loc[:,careas])
        # print(cur.columns)
        if self.fopt_plots['wind_curtailment'] and not self.fopt_no_plots :
            for a in cur.columns:
                f = self.get_fig()
                ax = f.subplots()
                xmax = (cur[a] < cur_thrsh).values.argmax()
                xmaxs.append(xmax)
                cur[a].plot(ax=ax)
                plt.xlim([0,xmax+10])
                plt.xlabel('Hours')
                plt.ylabel('GWh')
                if self.fopt_use_titles:
                    plt.title(f"{a} wind curtailment duration for case {self.name}")
                plt.grid()
                plt.tight_layout()

                plt.savefig(Path(self.fig_path) / Path(f"wcur_duration_{a}.png"))
                if self.fopt_eps:
                    plt.savefig(Path(self.fig_path) / Path(f"wcur_duration_{a}.eps"),dpi=self.fopt_dpi_qual)
                # plt.clf()
                

        if self.fopt_plots['wind_curtailment'] and not self.fopt_no_plots:
            # plot single figure with all curtailment curves
            # complement linestyles with width
            lss = []
            for i in range(clinewidths.__len__()):
                lss += clinestyles
            lws = []
            for i in range(clinewidths.__len__()):
                lws += [clinewidths[i] for s in clinestyles]
                #clinewidths.extend(clinewidths)

            # f,ax = plt.subplots()
            f = self.get_fig()
            ax = f.add_axes()
            f.set_size_inches(std_fig_size)
            for col,style,lw in zip(careas,lss+lss,lws+lws): # 16 styles should be enough
                cur[col].plot(style=style,lw=lw,ax=ax)
            plt.legend(careas)
            if xmaxs != []:
                plt.xlim([0,np.max(xmaxs)+10])

            plt.xlabel('Hours')
            plt.ylabel('GWh')
            if self.fopt_use_titles:
                plt.title(f'Wind curtailment duration curves for case {self.name}')
            plt.grid()
            plt.tight_layout()

            plt.savefig(Path(self.fig_path) / Path(f"wcur_duration_tot.png"))
            if self.fopt_eps:
                plt.savefig(Path(self.fig_path) / Path(f"wcur_duration_tot.eps"),dpi=self.fopt_dpi_qual)
            # plt.clf()
            

    def plot_miscellaneous(self):
        prt = self.opt_print['postprocess']


        if self.fopt_plots['losses'] and not self.fopt_no_plots:
            if prt:
                print('Plot losses')
            f = self.get_fig()
            self.res_losses.plot()
            plt.grid()
            plt.ylabel('Percent (%)')
            if self.fopt_use_titles:
                plt.title('{0}: Losses as percent of total internal generation'.format(self.name))
            compact_xaxis_ticks(plt.gcf(),plt.gca())
            plt.savefig(self.fig_path / 'losses.png')
            if self.fopt_eps:
                plt.savefig(self.fig_path / 'losses.eps',dpi=self.fopt_dpi_qual)
            # plt.clf()
            

        ## PLOT LOAD CURTAILMENT ##

        if self.fopt_plots['load_curtailment'] and not self.fopt_no_plots:
            if prt:
                print('Plot load curtailment')
            f = self.get_fig()
            ax = f.subplots()
            self.res_LS.plot(ax=ax)
            plt.grid()
            plt.ylabel('GW')
            if self.fopt_use_titles:
                plt.title('Load curtailment')
            compact_xaxis_ticks(plt.gcf(),plt.gca())
            plt.savefig(self.fig_path /  'load_curtailment.png')
            if self.fopt_eps:
                plt.savefig(self.fig_path /  'load_curtailment.eps',dpi=self.fopt_dpi_qual)
            


        ## PLOT INERTIA ##
        if self.fopt_plots['inertia'] and not self.fopt_no_plots:
            if prt:
                print('Plot inertia')
            f = self.get_fig()
            ax = f.subplots()
            self.res_ekin['model'].plot(ax=ax)
            if self.opt_use_inertia_constr:
                plt.plot([ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[-1]],[self.opt_min_kinetic_energy,self.opt_min_kinetic_energy],color='black',linestyle='--')
            plt.grid()
            plt.ylabel('GWs')
            if self.fopt_use_titles:
                plt.title('System kinetic energy')
            if self.opt_use_inertia_constr:
                plt.legend(['kinetic energy','min'])
            else:
                plt.legend(['kinetic energy'])

            compact_xaxis_ticks(plt.gcf(),ax)

            plt.savefig(self.fig_path /  'ekin_sys.png')
            if self.fopt_eps:
                plt.savefig(self.fig_path /  'ekin_sys.eps',dpi=self.fopt_dpi_qual)
            # plt.clf()
            

        if self.fopt_plots['hydro_duration'] and not self.fopt_no_plots:
            if prt:
                print('Plot hydro duration')
            for a in self.areas:
                if 'Hydro' in self.res_gen[a]:

                    if a in self.ror_areas:
                        hydro_model = duration_curve(self.res_gen[a].loc[:,['Hydro','HROR']].sum(axis=1))
                    else:
                        hydro_model = duration_curve(self.res_gen[a]['Hydro'])
                    hydro_data = MWtoGW*duration_curve(self.entsoe_data[a]['Hydro'])
                    f = self.get_fig()
                    ax = f.subplots()
                    hydro_model.plot(ax=ax)
                    hydro_data.plot(ax=ax)

                    plt.xlabel('Hours')
                    plt.ylabel('GWh')
                    if self.fopt_use_titles:
                        plt.title(f"Duration curve hydro {a}")
                    plt.grid()
                    plt.tight_layout()
                    plt.legend(['Model','Data'])
                    plt.savefig(Path(self.fig_path) / Path(f"hydro_duration_{a}.png"))
                    if self.fopt_eps:
                        plt.savefig(Path(self.fig_path) / Path(f"hydro_duration_{a}.eps"),dpi=self.fopt_dpi_qual)
                    # plt.clf()
                    

if __name__ == "__main__":

    def main_tag():
        pass

    pd.set_option('display.max_columns',20)
    year = 2019
    casename = f'test_paper_2'
    # casename = 'test'
    db_path = 'D:/Data/'
    m = Model(name=casename, path='D:/NordicModel/Runs', db_path=db_path,data_path='D:/NordicModel/InputData2')
    m.default_options()

    m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']
    # m.opt_countries = ['NO','FI','SE','DK']
    # m.opt_countries = ['NO','FI','SE','DK']

    # m.opt_start = '20191225'
    # m.opt_end = '20200107'
    m.opt_start = f'{year}0101'
    m.opt_end = f'{year}1231'
    m.opt_costfit_tag = f'{year}'
    m.opt_capacity_year = year
    m.opt_weather_year = year

    # m.opt_use_maf_pecd = True
    # m.opt_use_maf_inflow = True

    m.opt_solver = 'gurobi'
    m.opt_api = 'gurobi'

    # m.opt_use_var_exchange_cap = True
    # m.opt_run_initialization = True
    # m.opt_hydro_daily = False
    # m.opt_use_maf_inflow = False
    # m.opt_use_maf_pecd = False
    if year == 2016:
        m.vre_cap_2016()

    m.fopt_use_titles = True
    m.fopt_eps = False
    m.fopt_plots = {
        'gentype':True,
        'gentot':True,
        'gentot_bar':False,
        'transfer_internal':True,
        'transfer_external':True,
        'reservoir':True,
        'price':True,
        'losses':False,
        'load_curtailment':False,
        'inertia':False,
        'hydro_duration':False,
        'wind_curtailment':False,
    }
    m.fopt_calc_rmse = {
        'price':True,
        'transfer':True,
    }

    m.opt_print = {
        'init':True,
        'solver':True,
        'setup':True,
        'postprocess':True,
        'check':True,
    }

    m.fopt_no_plots = True
    m.run(save_model=True)

    # m.print_hydro_table()
    # # m.setup()
    # #
    # # RUN MODEL
    # # m.setup_child_model(api='pyomo')
    # # res = m.solve(solver='ipopt')
    # # m.post_process()
    # # m.save_model_results()
    #
    # # save run
    # m.save_model_run()
    # m.run_years(append=True,years=[2015,2016])




