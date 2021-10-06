"""
Create/modify databases for use in nordic_model

"""

import entsoe_transparency_db as entsoe
import pandas as pd
import maf_pecd_data
from pathlib import Path
from model_definitions import nordpool_areas, all_areas, all_areas_no_SE
# nordpool_areas = ('SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2','EE','LT','LV')
# all_areas = ('SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2',
#              'EE','LT','LV','GB','PL','NL','DE')
# all_areas_no_SE = ('NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2',
#                                'EE','LT','LV','GB','PL','NL','DE')

def price_database(startyear=2016,endyear=2019,areas=all_areas,db_name='prices.db',db_path='D:/Data'):

    db = entsoe.Database(db=Path(db_path) / db_name)
    db.download_price_data(areas=areas,startyear=startyear,endyear=endyear)


def exchange_database(startyear=2016,endyear=2020):
    db = entsoe.Database(db='D:/Data/exchange.db')

    #%
    db.define_connections(data_type='flow')
    # db.define_connections(data_type='exchange')
    db.define_connections(data_type='capacity')
    db.download_flow_data(nconnections=None,startyear=startyear,endyear=endyear,data_type='flow')
    db.download_flow_data(nconnections=None,startyear=startyear,endyear=endyear,data_type='capacity')

    # db.download_flow_data(data_type='capacity',nconnections=1,startyear=2020,endyear=2020)

    # from entsoe_transparency_db import add_nordpool_capacity_data
    # add_nordpool_capacity_data(startyear=2013,endyear=2020)

def gen_database(startyear=2016,endyear=2019,areas=all_areas_no_SE,db_name='gen.db',db_path='D:/Data'):

    db = entsoe.Database(db=Path(db_path)/db_name)
    db.download_gen_per_type_v2(startyear=startyear,endyear=endyear,areas=areas,intermediate_step=False)


def reservoir_database(startyear=2016,endyear=2019,db_name='reservoir.db',db_path='D:/Data',
       areas=['SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2','LV','LT','EE']):

    db = entsoe.Database(db=Path(db_path) / db_name)
    db.add_reservoir_data(areas=areas,start_year=startyear,end_year=endyear)

def inflow_data(db_name='inflow.db',table_name='inflow',fig_tag='c1',areas=[
        'NO1','NO2','NO3','NO4','NO5','FI','SE1','SE2','SE3','SE4','LT','LV'],week_start=4,proper_week=True):

    from entsoe_transparency_db import calculate_inflow_data_iso, calculate_inflow_data
    from week_conversion import WeekDef

    wd = WeekDef(week_start=week_start,proper_week=proper_week)


    starttime = '20141225:00'
    endtime = '20210105:00'
    # areas = ['NO1','NO2','NO3','NO4','NO5','FI'] + ['SE1','SE2','SE3','SE4'] + ['LT','LV']
    # areas = ['SE1','SE2','SE3','SE4']
    #
    startyear = 2015
    endyear = 2020

    from entsoe_transparency_db import validate_calculated_inflow_data

    for offset in [168]:
        # tab = f'{table_name}{offset}'
        tab = table_name
        inflow = calculate_inflow_data_iso(starttime,endtime,areas,offset=offset,table=tab,db_name=db_name,wd=wd)
        validate_calculated_inflow_data(inflow_table=tab,db=db_name,fig_tag=f'{fig_tag}_{offset}',wd=wd)

    return inflow

def load_database(startyear=2016,endyear=2019,areas=all_areas,db_name='load.db',db_path='D:/Data'):

    db = entsoe.Database(db=Path(db_path) / db_name)
    db.download_load_data(areas,startyear,endyear)

def maf_pecd_database(areas=['LV00','LT00','EE00']):
    db = maf_pecd_data.Database(db='D:/Data/entsoe_maf_pecd.db')

    print(f'--- ONSHORE 2025 ---')
    db.get_pecd_data(tab_type='onshore2025',areas=areas)
    print(f'--- OFFSHORE 2025 ---')
    db.get_pecd_data(tab_type='offshore2025',areas=areas)
    print(f'--- PV 2025 ---')
    db.get_pecd_data(tab_type='pv2025',areas=areas)


def cost_fit(years=range(2016,2020),areas=all_areas,db_path='D:/Data',path='D:/NordicModel/InputData'):

    from costfit import fit_shifted_costs
    # areas = ['SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2','LV','LT','EE','GB','DE','NL','PL']
    # areas = all_areas
    for year in years:
        fit_shifted_costs(areas=areas,starttime=f'{year}0101:00',endtime=f'{year}1231:23',period_plots=False,tag=f'{year}',
                          path=Path(path)/'costfit',fig_title=False,db_path=db_path)

        # fit_shifted_costs(areas=areas,starttime='20160101:00',endtime='20161231:23',period_plots=True,tag='2016',path=path)
        # fit_shifted_costs(areas=areas,starttime='20190101:00',endtime='20191231:23',period_plots=True,tag='2019',path=path)

def capacity_database(startyear=2016,endyear=2020,areas=all_areas,db_name='capacity.db',db_path='D:/Data'):

    db = entsoe.Database(Path(db_path) / db_name)
    db.download_cap_per_type_data(start_year=startyear,end_year=endyear,areas=areas)

def unit_database(startyear=2016,endyear=2020,countries=['SE','FI','DK'],db_name='unit.db',db_path='D:/Data'):
    db = entsoe.DatabaseGenUnit(Path(db_path) / db_name)
    db.download_data(starttime=f'{startyear}0101',endtime=f'{endyear}1231',countries=countries)

def calculate_excel_data(data_path='D:/NordicModel/Data',db_path='D:/Data'):
    """
    Generate excel files with model parameters.

    load_shares - used for distributing solar capacity for countries with multiple price areas if capacities
    are specified on per country basis

    gen_stats - max/min yearly production values from entsoe data, used for capacities in model

    map_NO_maf_areas - make mapping between maf areas and NO price areas, to distribute maf inflow to different
    price zones (since MAF uses 3 areas for NO, different from the 5 price zones)

    """
    from statistic_calculations import compute_load_shares, get_entsoe_production_stats
    from maf_hydro_data import map_NO_maf_areas

    Path(data_path).mkdir(exist_ok=True,parents=True)

    # load_shares.xlsx
    compute_load_shares(year=2020,data_path=data_path,db_path=db_path)

    # gen_stats.xlsx
    get_entsoe_production_stats(startyear=2016,endyear=2020,areas=all_areas,data_path=data_path,db_path=db_path)

    # maf_to_bidz_map.xlsx
    map_NO_maf_areas(db_path=db_path,data_path=data_path)

    # costfit
    cost_fit(years=range(2016,2021))

def create_databases(startyear=2016,endyear=2020,db_path='D:/Data'):
    """
    Script to download additional data needed to run model.
    Note: To run the access token must be obtained from ENTSO-E Transparency Platform by registering, and
    entered in line 3 of entsoe_transparency_db.py
    To retrieve a token, see the instructions on:
    https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
    """
    # gen capacity data: capacity.db
    print('----- DOWNLOADING GEN CAPACITY DATA ----')
    capacity_database(startyear=startyear,endyear=endyear,db_path=db_path)

    # reservoir data: reservoir.db
    print('----- DOWNLOADING RESERVOIR DATA ----')
    reservoir_database(startyear=startyear,endyear=endyear,db_path=db_path)

    # prices: prices.db
    print('----- DOWNLOADING PRICE DATA ----')
    price_database(startyear=startyear,endyear=endyear,db_path=db_path)

    # load: load.db
    print('----- DOWNLOADING LOAD DATA ----')
    load_database(startyear=startyear,endyear=endyear,db_path=db_path)

    print('----- DOWNLOADING GENERATION PER TYPE DATA ----')
    # gen: gen.db (Swedish data already present)
    gen_database(startyear=startyear,endyear=endyear,db_path=db_path)

    print('----- DOWNLOADING GENERATION PER UNIT DATA ----')
    # unit: unit.db
    unit_database(startyear=startyear,endyear=endyear,db_path=db_path,countries=['SE','FI','DK'])

if __name__ == "__main__":

    pd.set_option('display.max_rows',100)
    pd.set_option('display.max_columns',None)

    create_databases(db_path='D:/Data/Model Release')

