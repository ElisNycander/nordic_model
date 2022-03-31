# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:27:07 2019

@author: elisn
"""

from nordic_model import Model
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from help_functions import str_to_date
from model_definitions import MWtoGW


db_path = 'D:/Data/Model Release'
data_path = 'D:/NordicModel/InputData3'

def solve_benchmark_cases(ncases=2,path='D:/NordicModel/Runs'):

    for case in ['ipopt','ipopt_ini','gurobi']:
        for i in range(ncases):
            casename = f'time_{case}_{i}'
            m = Model(name=casename, path=path, db_path='D:/Data/',data_path='D:/NordicModel/InputData')
            m.default_options()

            m.opt_solver = case.split('_')[0]
            m.opt_api = 'gurobi'
            if case == 'ipopt_ini':
                m.opt_run_initialization = True
            else:
                m.opt_run_initialization = False

            ###### remaining options, same for all cases #######

            m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']

            m.opt_start = '20190101'
            m.opt_end = '20191230'
            m.opt_use_var_exchange_cap = True
            m.opt_nominal_capacity_connections = [('GB','NL')]
            m.opt_use_var_cost = True
            m.opt_hydro_cost = False
            m.opt_costfit_tag = '2019'
            m.opt_pmax_type = 'capacity'
            m.opt_pmax_type_hydro = 'stats'
            m.opt_use_reserves = True
            m.opt_country_reserves = False
            m.opt_solar_source = 'entsoe'
            m.opt_set_external_price = ['DE','PL']
            m.opt_hydro_daily = False
            m.opt_use_maf_inflow = False
            m.opt_weather_year = 2019
            m.opt_use_maf_pecd = False

            m.fopt_plots = {
                'gentype':False,
                'gentot':False,
                'gentot_bar':False,
                'transfer_internal':False,
                'transfer_external':False,
                'reservoir':False,
                'price':False,
                'losses':False,
                'load_curtailment':False,
                'inertia':False,
                'hydro_duration':False,
                'wind_curtailment':False,
            }
            m.fopt_calc_rmse = {
                'price':False,
                'production':False,
                'transfer':False,
            }

            m.opt_print = {
                'init':True,
                'solver':True,
                'setup':True,
                'postprocess':True,
                'check':True,
            }

            m.run()

def compare_benchmark_cases(ncases=2,path = 'D:/NordicModel/Runs'):
    #%% compare performance benchmark cases
    fig_path = 'D:/NordicModel/Figures'
    ncases = 1

    tdic = {}
    for i in range(ncases):
        for case in ['gurobi','ipopt','ipopt_ini']:
            if case not in tdic:
                tdic[case] = pd.DataFrame(index=range(ncases),columns=['solver','cm','tot'])
            m = Model(name=f'time_{case}_{i}',path=path)
            m.load_model_results()

            df = tdic[case]
            for c in df.columns:
                df.at[i,c] = m.res_time[c]

    for case in tdic:
        df = tdic[case]
        df['other'] = df['tot'] - df['solver'] - df['cm']

    # compute averages
    df = pd.DataFrame(columns=list(tdic.keys()),index=['solver','cm','other'])
    for i in df.index:
        for c in df.columns:
            df.at[i,c] = tdic[c][i].mean()
    df.index = ['solver','setup','other']
    df = df/60

    #%%
    df = df.transpose()
    # plot bar chart
    f,ax = plt.subplots()
    f.set_size_inches(5.5,3.5)
    # df.transpose().plot.bar(stacked=True,ax=ax,rot=0)
    plt.bar(x=df.index,height=df['solver'],color='C0',label='solver',hatch='xx')
    plt.bar(x=df.index,height=df['setup'],bottom=df['solver'],color='C1',label='setup')
    plt.bar(x=df.index,height=df['other'],bottom=df['setup']+df['solver'],color='C2',label='other',hatch='||')
    plt.legend()
    # (df['setup']+df['solver']).plot(bottom=df['solver'],color='C1')

    plt.ylabel('Time (min)')
    plt.grid()
    plt.savefig(Path(fig_path) / f'model_benchmark.png')
    plt.show()

    return df

def solve_benchmark_cases_multiyear(tag='time',ncases=2,path='D:/NordicModel/Runs',years=range(1990,1992)):

    for case in ['ipopt','ipopt_ini','gurobi']:
        for i in range(ncases):
            casename = f'{tag}_{case}_{i}'
            print(f'------- {casename} ----------')

            m = Model(name=casename, path=path, db_path='D:/Data/',data_path='D:/NordicModel/InputData')
            m.default_options()

            m.opt_solver = case.split('_')[0]
            m.opt_api = 'gurobi'
            if case == 'ipopt_ini':
                m.opt_run_initialization = True
            else:
                m.opt_run_initialization = False

            ###### remaining options, same for all cases #######

            m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']

            m.opt_start = '20160101'
            m.opt_end = '20161231'
            m.opt_use_var_exchange_cap = True
            m.opt_nominal_capacity_connections = [('GB','NL')]
            m.opt_use_var_cost = True
            m.opt_hydro_cost = False
            m.opt_costfit_tag = '2016'
            m.opt_pmax_type = 'capacity'
            m.opt_pmax_type_hydro = 'stats'
            m.opt_use_reserves = True
            m.opt_country_reserves = False
            m.opt_solar_source = 'entsoe'
            m.opt_set_external_price = ['DE','PL']
            m.opt_hydro_daily = False
            m.opt_use_maf_inflow = True
            m.opt_weather_year = 2016
            m.opt_use_maf_pecd = True

            m.fopt_plots = {
                'gentype':False,
                'gentot':False,
                'gentot_bar':False,
                'transfer_internal':False,
                'transfer_external':False,
                'reservoir':False,
                'price':False,
                'losses':False,
                'load_curtailment':False,
                'inertia':False,
                'hydro_duration':False,
                'wind_curtailment':False,
            }
            m.fopt_calc_rmse = {
                'price':False,
                'production':False,
                'transfer':False,
            }

            m.opt_print = {
                'init':False,
                'solver':False,
                'setup':False,
                'postprocess':False,
                'check':True,
            }

            m.run_years(years=years)

def compare_benchmark_cases_multiyear(tag='time',years = [2015,2016],cases=['gurobi','ipopt','ipopt_ini'],nruns=2,
                                      fig_path='D:/NordicModel/Figures',path='D:/NordicModel/Runs'):
    """
    Compare cases when model is run for multiple years
    :return:
    """
    #%% calculate runtime for model which has been run multiple years
    # fig_path = 'D:/NordicModel/Figures'
    # path = 'D:/NordicModel/Runs'
    # years = [2015,2016]
    # cases = ['test_update']
    # nruns = 1
    ncases = cases.__len__()
    nyears = years.__len__()


    # time_dic = {}
    time_df = pd.DataFrame(0.0,dtype=float,
                           index=pd.MultiIndex.from_product([cases,['base','add'],],names=['case','year']),
                           columns=['post','solver','cm','pre'])
    for case in cases:
        # time_df = pd.DataFrame(0.0,dtype=float,index=['base','add'],
        #                        columns=['solver','cm','tot'])
        for ridx in range(nruns):
            for year in years:
                casename = f'{tag}_{case}_{ridx}'
                m = Model(path=Path(path),name=casename)
                m.load_model_run(year)
                if year == years[0]: # base case
                    for c in time_df.columns:
                        time_df.at[(case,'base'),c] += m.res_time[c] / nruns
                else: # additional year
                    for c in time_df.columns:
                        time_df.at[(case,'add'),c] += m.res_time[c] / (nyears-1) / nruns

    # #% make plot
    time_df.columns = ['post-process','solver','setup','pre-process',]
    time_df.index = pd.MultiIndex.from_product([cases,['base','re-run'],],names=['solver','case'])
    df = time_df.transpose() / 60

    #%% make plot
    # df = df.transpose()
    # plot bar chart
    f,ax = plt.subplots()
    if cases.__len__() == 2:
        f.set_size_inches(6,4.5)
    else:
        f.set_size_inches(10,4.5)
    df.transpose().plot.bar(stacked=True,ax=ax,rot=0,linewidth=0.8,edgecolor='black')

    bars = ax.patches
    from itertools import cycle
    hatches = ['','///','xxx','|||']
    all_hatches = []
    for h in hatches:
        for i in range(ncases*2):
            all_hatches.append(h)

    for hatch,bar in zip(all_hatches,bars):
        bar.set_hatch(hatch)
    plt.legend()
    plt.ylabel('Time (min)')
    plt.grid()
    plt.savefig(Path(fig_path) / f'model_benchmark_years.png')
    plt.savefig(Path(fig_path) / f'model_benchmark_years.eps')
    plt.show()
    return  df

def run_DE_pump_hydro_cases(path='D:/NordicModel/Runs'):

    casenames = ['germany_default','germany_maf_pump']
    for name in casenames:
    
        year = 2019
        # db_path = 'D:/Data/'
        m = Model(name=name, path=path, db_path=db_path,data_path=data_path)
        m.default_options()
        
        if name == 'germany_maf_pump': # use maf pump reservoir capacity
            m.opt_pump_reservoir['DE'] = 608
            m.opt_reservoir_capacity['DE'] = 675

        # all other options same
        m.opt_start = f'{year}0416'
        m.opt_end = f'{year}0615'
        m.opt_costfit_tag = f'{year}'
        m.opt_capacity_year = year
    
        m.opt_solver = 'gurobi'
        m.opt_api = 'gurobi'
        m.opt_hydro_daily = False
        m.opt_use_maf_inflow = False
        m.opt_weather_year = year
        m.opt_use_maf_pecd = False
    
        m.fopt_use_titles = False
        m.fopt_eps = True
        m.fopt_plots = {
            'gentype':True,
            'gentot':False,
            'gentot_bar':False,
            'transfer_internal':False,
            'transfer_external':False,
            'reservoir':True,
            'price':False,
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
    
        m.run(save_model=True)

def run_hydro_resolution_cases(path='D:/NordicModel/Runs'):

    cases = ['nordic_hourly','nordic_daily']

    year = 2019

    for name in cases:

        # db_path = 'D:/Data/'
        m = Model(name=name, path=path, db_path=db_path,data_path=data_path)
        m.default_options()

        if name == 'nordic_hourly':
            m.opt_hydro_daily = False
        else:
            m.opt_hydro_daily = True

        m.opt_countries = ['NO','FI','SE','DK']

        m.opt_start = f'{year}0101'
        m.opt_end = f'{year}1230'
        m.opt_costfit_tag = f'{year}'
        m.opt_capacity_year = year

        m.opt_solver = 'gurobi'
        m.opt_api = 'gurobi'
        m.opt_use_maf_inflow = False
        m.opt_weather_year = year
        m.opt_use_maf_pecd = False
        m.fopt_eps = False
        m.fopt_plots = {
            'gentype':False,
            'gentot':False,
            'gentot_bar':False,
            'transfer_internal':False,
            'transfer_external':False,
            'reservoir':False,
            'price':False,
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
        m.run(save_model=True)

def compare_hydro_resolution_cases(path='D:/NordicModel/Runs',fig_path='D:/NordicModel/Figures'):

    #%% compare hourly/daily reservoir modelling

    # path = 'D:/NordicModel/Runs'
    # fig_path = 'D:/NordicModel/Figures'
    fig_name = 'hydro_hourly_vs_daily'

    m1 = Model(path=path,name='nordic_hourly')
    m2 = Model(path=path,name='nordic_daily')
    m1.load_model()
    m2.load_model()

    #%%
    area = 'SE1'
    from help_functions import interp_time
    import entsoe_transparency_db
    db = entsoe_transparency_db.Database(db='D:/Data/reservoir.db')
    res_df = db.select_reservoir_wrap(areas=[area],starttime=m1.opt_start+':00',endtime=m1.opt_end+':23')
    res_df_interp = interp_time(dates=m1.res_RES.index,df=res_df)

    #%%
    f,ax = plt.subplots()
    f.set_size_inches(6,4)

    m1.res_RES[area].plot(ax=ax,label='hourly')
    m2.res_RES[area].plot(ax=ax,label='daily',linestyle='--')
    (res_df_interp[area]/1e3).plot(ax=ax,label='ENTSO-E',linestyle='-',color='k',linewidth=0.7)


    plt.grid()
    plt.legend()
    plt.ylabel('Reservoir content (TWh)')
    plt.savefig(Path(fig_path) / f'{fig_name}.eps')
    plt.savefig(Path(fig_path) / f'{fig_name}.png')

    #%% table for comparison

    df = pd.DataFrame(dtype=float,columns=['daily','weekly','diff'],index=['obj','nvars','nconstr'])


    for m,c in zip([m1,m2],['daily','weekly']):
        df.at['obj',c] = m.res_obj
        df.at['nvars',c] = m.res_stats['NumVars']
        df.at['nconstr',c] = m.res_stats['NumConstrs']

    df['diff'] = (df['weekly']-df['daily'])/df['daily']*1e2
    # for f in df.index:
    #     if f
    df.columns = ['Daily','Weekly','Diff (%)']
    table_name = 'hydro_compare.txt'

    with open(Path(fig_path)/table_name,'wt') as f:

        df.transpose().to_latex(f,header=['Objective','NumVars','NumConstrs'],)

def validate_maf_wind_solar():


    def main_tag():
        pass

    pd.set_option('display.max_columns',20)
    year = 2016
    casename = f'maf_validation'
    # casename = 'test'
    db_path = 'D:/Data/'
    m = Model(name=casename, path='D:/NordicModel/Runs', db_path=db_path,data_path='D:/NordicModel/InputData')
    m.default_options()

    m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']

    m.opt_start = '20160101'
    m.opt_end = '20160630'

    m.opt_costfit_tag = f'{year}'
    m.opt_capacity_year = year

    m.opt_use_var_exchange_cap = False
    m.opt_solver = 'gurobi'
    m.opt_api = 'gurobi'
    m.opt_hydro_daily = False
    m.opt_use_maf_inflow = False
    m.opt_weather_year = year
    # m.opt_weather_year = 2019
    m.opt_use_maf_pecd = True
    if year == 2016:
        m.vre_cap_2016()

    m.fopt_use_titles = False
    m.fopt_eps = False
    m.fopt_plots = {
        'gentype':True,
        'gentot':False,
        'gentot_bar':False,
        'transfer_internal':False,
        'transfer_external':False,
        'reservoir':True,
        'price':False,
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

    m.setup_indices()
    m.setup_weather_indices()
    m.setup_transmission()
    m.setup_data()
    m.setup_solar()
    m.setup_wind()

    #%% compare MAF and ENTSO-E wind and solar
    use_norm = False


    for gtype in ['Wind','Solar']:
        f,ax = plt.subplots()

        if gtype == 'Wind':
            iter_areas = [a for a in m.wind_areas if gtype in m.entsoe_data[a].columns]
        else:
            iter_areas = [a for a in m.solar_areas if gtype in m.entsoe_data[a].columns]
        for area in iter_areas:
            if gtype == 'Wind':
                df_maf = m.wind[area]
            else:
                df_maf = m.solar[area]

            df_tp = m.entsoe_data[area][gtype]/1e3
            if use_norm:
                df_maf = df_maf / df_maf.max()
                df_tp = df_tp / df_tp.max()

            f.clf()
            f.set_size_inches(4,6)
            ax = f.add_subplot(1,1,1)
            df_maf.plot(ax=ax,label='MAF')
            df_tp.plot(ax=ax,label='ENTSO-E TP')

            rmse = np.sqrt(np.mean(np.square(df_maf-df_tp)))
            plt.grid()
            plt.legend(title=f'RMSE: {rmse:0.2f}')
            plt.title(area)
            plt.ylabel('GWh')
            plt.savefig(m.fig_path / f'{gtype}_rmse_{area}.png')


def validation(year=2019,path='D:/NordicModel/Runs'):
    #%% 2019 validation


    pd.set_option('display.max_columns',20)
    # year = 2019
    casename = f'validate_{year}'
    # casename = 'test'
    # db_path = 'D:/Data/'
    m = Model(name=casename, path=path, db_path=db_path,data_path=data_path)
    m.default_options()

    m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']

    m.opt_start = f'{year}0101'
    m.opt_end = f'{year}1230'
    m.opt_costfit_tag = f'{year}'
    m.opt_capacity_year = year
    m.opt_weather_year = year

    m.opt_solver = 'gurobi'
    m.opt_api = 'gurobi'

    m.fopt_use_titles = False
    m.fopt_eps = True
    # m.fopt_plots = {
    #     'gentype':True,
    #     'gentot':True,
    #     'gentot_bar':False,
    #     'transfer_internal':True,
    #     'transfer_external':True,
    #     'reservoir':True,
    #     'price':True,
    #     'losses':False,
    #     'load_curtailment':False,
    #     'inertia':False,
    #     'hydro_duration':False,
    #     'wind_curtailment':False,
    # }
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

    m.run(save_model=True)

    return m

def case_study_example(years=range(1982,2017),path='D:/NordicModel/Runs'):
    """ illustrative case study """
    pass

    pd.set_option('display.max_columns',20)
    year = 2018
    casename = f'casestudy_{year}'
    # casename = 'test'
    # db_path = 'D:/Data/'
    m = Model(name=casename, path=path, db_path=db_path,data_path=data_path)
    m.default_options()

    m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']
    # m.opt_countries = ['NO','FI','SE','DK']
    # m.opt_countries = ['NO','FI','SE','DK']

    # m.opt_start = '20160101'
    # m.opt_end = '20160630'
    m.opt_start = f'{year}0101'
    m.opt_end = f'{year}1231'
    m.opt_costfit_tag = f'{year}'
    m.opt_capacity_year = year
    m.opt_weather_year = 2016
    m.opt_use_maf_pecd = True
    m.opt_use_maf_inflow = True

    m.opt_solver = 'gurobi'
    m.opt_api = 'gurobi'

    #% case study settings
    m.opt_load_scale = 1.064
    m.opt_nucl_individual_units = ['SE3']
    m.opt_nucl_units_exclude = ['Ringhals block 1 G11','Ringhals block 1 G12','Ringhals block 2 G21','Ringhals block 2 G22']
    m.opt_exchange_cap_year = 2025
    m.opt_nucl_add_cap = {
        'SE3':0,
        'FI':1600*0.85,
    }
    m.opt_nucl_min_lvl = 0.65

    update_solar_cap_by_country = { # upate solar, same as table 10, except SE which has 2 GW
        'NO':500,
        'SE':2000,
        'FI':500,
        'DK':1500,
    }
    from model_definitions import country_to_areas
    for c,val in update_solar_cap_by_country.items():
        m.opt_solar_cap_by_country[c] = val
        for a in country_to_areas[c]:
            if a in m.opt_solar_cap_by_area:
                del m.opt_solar_cap_by_area[a]

    # update wind power, same capacities as table 9 [Nycander 2020]
    update_wind_capacity_onsh = {
        'DK1':5066-1277,
        'DK2':1689-423,
        'FI':3442,
        'NO1':200,
        'NO2':1202,
        'NO3':1573,
        'NO4':745,
        'NO5':0,
        'SE1':2516,
        'SE2':5113,
        'SE3':3230,
        'SE4':1594,
    }
    for a,val in update_wind_capacity_onsh.items():
        m.opt_wind_capacity_onsh[a] = val


    m.fopt_use_titles = True
    m.fopt_eps = False
    m.fopt_plots = {
        'gentype':True,
        'gentot':True,
        'gentot_bar':True,
        'renewables':True,
        'transfer_internal':True,
        'transfer_external':True,
        'reservoir':True,
        'price':True,
        'losses':False,
        'load_curtailment':False,
        'inertia':False,
        'hydro_duration':False,
        'wind_curtailment':True,
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

    m.run_years(years=years,save_full_model=True)

def case_study_results(name='casestudy_2018',years=range(1982,2000),path='D:/NordicModel/Runs'):

    m = Model(name=name,path=path)

    res = m.load_results_years(vars=['res_cur_WIND','res_WIND'],years=years)

    #%% plot renewable curtailment
    import datetime
    areas = ['SE1','SE2','SE3','SE4','DK1','DK2']
    years = list(res[list(res.keys())[0]].keys())
    df = pd.DataFrame(dtype=float,index=years,columns=areas)
    cur_tot = pd.Series(dtype=float,index=years)

    for y in years:
        for a in areas:
            df.at[y,a] = res['res_cur_WIND'][y][a].sum() / (res['res_cur_WIND'][y][a].sum() + res['res_WIND'][y][a].sum()) * 1e2
            cur_tot[y] = (res['res_cur_WIND'][y].sum().sum()) / (res['res_cur_WIND'][y].sum().sum() \
                                                                 + res['res_WIND'][y].sum().sum()) * 1e2

    #%% histogram with curtailment
    f,ax = plt.subplots()

    # bars = cur_tot.plot.bar(ax=ax)
    cur_tot.plot(ax=ax,linestyle='none',marker='o',markersize=8)
    plt.xlim([cur_tot.index.min()-0.5,cur_tot.index.max()+0.5])
    plt.grid()
    # plot average
    avg = cur_tot.mean()
    plt.plot(ax.get_xlim(),[avg,avg],linestyle='--',color='k',label='avg')
    plt.legend(['yearly','avg'])
    # plt.legend(['avg','yearly'])
    plt.ylabel('Curtailment (%)')
    plt.ylim([0,plt.ylim()[1]])

    plt.savefig(m.fig_path / f'case_study.png')
    plt.savefig(m.fig_path / f'case_study.eps')

def paper_results(path='D:/NordicModel/PaperRuns',fig_path='D:/NordicModel/PaperFigures'):

    Path(fig_path).mkdir(exist_ok=True,parents=True)
    from create_databases import cost_fit, create_databases
    """ All runs for paper """

    from entsoe_transparency_db import validate_inflow_data, compare_maf_ninjas_entsoe
    validate_inflow_data(fig_path=fig_path)
    compare_maf_ninjas_entsoe(fig_path=fig_path)

    print('-------- COST FIT ----------')
    cost_fit(years=range(2016,2020))
    create_databases()


    print('-------- PUMP HYDRO CASES ----------')
    run_DE_pump_hydro_cases(path=path)

    print('-------- RESERVOIR RESOLUTION CASES ----------')
    run_hydro_resolution_cases(path=path)
    compare_hydro_resolution_cases(path=path,fig_path=fig_path)

    print('-------- VALIDATION CASE ----------')
    validation(year=2019,path=path)

    print('-------- CASE STUDY CASES ----------')
    # case study
    case_study_example(years=range(1982,2017),path=path)
    case_study_results(name='casestudy_2018',path=path,years=None)


    print('-------- BENCHMARK CASES ----------')
    ncases = 5
    tag = 'time'
    years = range(1990,1993)
    # solve_benchmark_cases_multiyear(tag=tag,ncases=ncases,years=years,path=path)
    df = compare_benchmark_cases_multiyear(tag=tag,nruns=ncases,years=years,cases=['gurobi','ipopt'],
                                           path=path,fig_path=fig_path)
    return df

def run_cost_compare_cases():
    """ Run cases to compare using constant and shifting marginal costs """

    import matplotlib.dates as mdates

    path = 'D:/NordicModel/ThesisRuns'
    fig_path = 'D:/NordicModel/Figures'
    area = 'DK1'
    gen = 21
    year = 2018

    #%% compare runs with weekly/constant marginal cost offset

    for tag in ['mw','mavg']:
        casename = f'mcompare_{year}_{tag}'
        # casename = 'test'
        # db_path = 'D:/Data/'
        m = Model(name=casename, path=path, db_path=db_path,data_path=data_path)
        m.default_options()

        if tag == 'mw':
            m.opt_use_var_cost = True
        else:
            m.opt_use_var_cost = False

        m.opt_countries = ['NO','FI','SE','DK','LV','EE','LT','PL','DE','NL','GB']

        m.opt_start = f'{year}0101'
        m.opt_end = f'{year}1230'
        m.opt_costfit_tag = f'{year}'
        m.opt_capacity_year = year
        m.opt_weather_year = year

        m.opt_solver = 'gurobi'
        m.opt_api = 'gurobi'

        m.fopt_use_titles = False
        m.fopt_eps = False
        m.fopt_plots = {
            'gentype':True,
            'gentot':True,
            'gentot_bar':False,
            'transfer_internal':False,
            'transfer_external':False,
            'reservoir':True,
            'price':False,
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

        m.run(save_model=True)



    #%% compare thermal generation for different runs

    m1 = Model(name='mcompare_2018_mw',path=path)
    m1.load_model()

    m2 = Model(name='mcompare_2018_mavg',path=path)
    m2.load_model()

    #%%

    inset_days = 8
    inset_date = '20180401:00'
    inset_idx = pd.date_range(start=str_to_date(inset_date),
                              end=str_to_date(inset_date)+datetime.timedelta(days=inset_days),
                              freq='H')
    ax2_width = 0.6
    # legend position with inset
    bbox = (0.99,1.5) # horizontal, vertical
    inset_height_ratio = [1,1.6]
    myFmt = '%m-%d'

    # f,ax = plt.subplots()
    f = plt.figure()
    ax1,ax2 = f.subplots(2,1,gridspec_kw={'height_ratios':inset_height_ratio})
    pos = ax1.get_position()
    ax1.set_position([pos.x0,pos.y0,ax2_width,pos.height])

    #% divide figure


    #% main plot
    m1.res_PG[gen].plot(ax=ax2,label='m',color='C0')
    m2.res_PG[gen].plot(ax=ax2,label='mavg',color='C2')

    (m.entsoe_data[area]['Thermal']*MWtoGW).plot(ax=ax2,label='data',color='C1',linestyle='dashed',linewidth=0.9)

    ax2.set_ylabel('GWh')
    ax2.set_zorder(1)
    ax2.grid()
    # remove line breaks from ticks
    from help_functions import compact_xaxis_ticks
    compact_xaxis_ticks(f,ax2)
    # ax1.legend([],title=self.plot_vals.annstr.format(irmse_rel,irmse),bbox_to_anchor=self.plot_vals.bbox)
    ax2.legend(bbox_to_anchor=bbox)


    #% inset
    ax1.plot(m1.res_PG.loc[inset_idx,gen],label='m',color='C0')
    ax1.plot(m2.res_PG.loc[inset_idx,gen],label='mavg',color='C2')
    ax1.plot(m.entsoe_data[area].loc[inset_idx,'Thermal']*MWtoGW,label='data',color='C1',linestyle='dashed',linewidth=0.9)

    ax1.grid()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(myFmt))
    ax1.set_ylabel('GWh')
    ax1.xaxis.set_major_locator(mdates.DayLocator())

    plt.savefig(Path(fig_path) / f'cost_compare_{area}.png')
    plt.savefig(Path(fig_path) / f'cost_compare_{area}.eps')

def plot_cost_compare_cases():

    import matplotlib.dates as mdates
    path = 'D:/NordicModel/ThesisRuns'
    fig_path = 'D:/NordicModel/Figures'
    area = 'DK1'
    gen = 21
    year = 2018
    starttime = f'{year}0101'
    endtime = f'{year}1230'
    cet = False
    #%% compare thermal generation for different runs

    m1 = Model(name='mcompare_2018_mw',path=path)
    m1.load_model()

    m2 = Model(name='mcompare_2018_mavg',path=path)
    m2.load_model()

    from entsoe_transparency_db import Database

    db = Database('D:/Data/gen.db')
    gen_dic = db.select_gen_per_type_wrap_v2(starttime=starttime,endtime=endtime,areas=[area],cet_time=cet)

    #%%

    inset_days = 8
    inset_date = '20180401:00'
    inset_idx = pd.date_range(start=str_to_date(inset_date),
                              end=str_to_date(inset_date)+datetime.timedelta(days=inset_days),
                              freq='H')
    ax2_width = 0.6
    # legend position with inset
    bbox = (1.06,1.75) # horizontal, vertical
    inset_height_ratio = [1,1.6]
    myFmt = '%m-%d'

    # f,ax = plt.subplots()
    f = plt.figure()
    ax1,ax2 = f.subplots(2,1,gridspec_kw={'height_ratios':inset_height_ratio})
    pos = ax1.get_position()
    ax1.set_position([pos.x0,pos.y0,ax2_width,pos.height])

    #% divide figure


    #% main plot
    m1.res_PG[gen].plot(ax=ax2,label='m (weekly)',color='C0')
    m2.res_PG[gen].plot(ax=ax2,label='m (avg)',color='C2')

    (gen_dic[area]['Thermal']*MWtoGW).plot(ax=ax2,label='data',color='C1',linestyle='dashed',linewidth=0.9)

    nmae1 = (m1.res_PG[gen]-gen_dic[area]['Thermal']*MWtoGW).abs().mean()/(gen_dic[area]['Thermal'].mean()*MWtoGW)
    nmae2 = (m2.res_PG[gen]-gen_dic[area]['Thermal']*MWtoGW).abs().mean()/(gen_dic[area]['Thermal'].mean()*MWtoGW)

    ax2.set_ylabel('GWh')
    ax2.set_zorder(1)
    ax2.grid()
    # remove line breaks from ticks
    from help_functions import compact_xaxis_ticks
    compact_xaxis_ticks(f,ax2)
    # ax1.legend([],title=self.plot_vals.annstr.format(irmse_rel,irmse),bbox_to_anchor=self.plot_vals.bbox)
    ax2.legend(bbox_to_anchor=bbox,title=f'NMAE: \nweekly: {nmae1:0.2f}\navg: {nmae2:0.2f}')


    #% inset
    ax1.plot(m1.res_PG.loc[inset_idx,gen],label='m',color='C0')
    ax1.plot(m2.res_PG.loc[inset_idx,gen],label='mavg',color='C2')
    ax1.plot(gen_dic[area].loc[inset_idx,'Thermal']*MWtoGW,label='data',color='C1',linestyle='dashed',linewidth=0.9)

    ax1.grid()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(myFmt))
    ax1.set_ylabel('GWh')
    ax1.xaxis.set_major_locator(mdates.DayLocator())

    plt.savefig(Path(fig_path) / f'cost_compare_{area}.png')
    plt.savefig(Path(fig_path) / f'cost_compare_{area}.eps')


if __name__ == "__main__":
    # plot_cost_compare_cases()
    # df = paper_results()
    pass

    # run_cost_compare_cases()
    # m = validation(year=2019,path='D:/NordicModel/PaperRuns')

