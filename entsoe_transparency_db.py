# -*- coding: utf-8 -*-
###### PERSONAL ACCESS TOKEN FOR ENTSO-E TRANSPARENCY MUST BE DEFINED ########
try:
    with open('token.txt','rt') as f:
            req_token = f.readline()
    if req_token == '':
        raise Exception("WARNING: Missing access token for transparency platform in 'token.txt'")
except FileNotFoundError:
    raise Exception("WARNING: Add file 'token.txt' with access token for transparency platform to repository")

""" Database with ENTSO-E transparency data. Data is downloaded 
    in xml format from the transparency server using a required
    key. This module also has some functions for processing the 
    transparency data. 

    Tables:
    
    cap_per_type(TEXT year, TEXT area, TEXT type, FLOAT cap)
        - installed capacities, note that data for SE regions is missing
        from entsoe database
        
    gen_per_type(TEXT time, TEXT area, TEXT type, FLOAT gen)
        - actual generation for different production types from 
        2015-2018. 93 MB of data, takes ~6 hours to download
        
    se_gen_per_type(TEXT time, TEXT area, TEXT type, FLOAT gen)
        - generation per production type for SE, from SvK data
        Has broader categories: Wind, Solar, Nuclear, Hydro, CHP, Gas, Other
        
    gen_per_unit - actual generation per unit, not implemented
    
    Note: Entso-e transparency data also has installed capacity 
    per type and per unit. This data can be downloaded using 
    get_entsoe_gen_data() with datatype=1/2. However, since the amount of
    data is small it is not stored in the sqlite database, as it
    can be downloaded directly when it is needed.
    
    Note: For nordpool data for hour 00-01 we have used time stamp YYYYMMDD:00
    ENTSO-E data in UTC, Nordpool data in CET=UTC+1:
    Nordpool  -> Entsoe
    20180101:00  20171231:23
    20180101:01  20180101:00
    

    
Created on Wed Jan 16 11:31:07 2019

@author: elisn
"""

"""
Process Type:
    A01 - Day ahead
    A02 - Intra day incremental
    A16 - Realised 
    A18 - Intraday total 
    A31 - Week ahead 
    A32 - Month ahead 
    A33 - Year ahead 
    A39 - Synchronization process
    A40 - Intraday process

Document Type:
    A09 - Finalised schedule
    A11 - Aggregated energy data report
    A25 - Allocation result document
    A26 - Capacity document
    A31 - Agreed capacity
    A44 - Price Document
    A61 - Estimated Net Transfer Capacity
    A63 - Redispatch notice
    A65 - System total load
    A68 - Installed generation per type
    A69 - Wind and solar forecast
    A70 - Load forecast margin
    A71 - Generation forecast
    A72 - Reservoir filling information
    A73 - Actual generation
    A74 - Wind and solar generation
    A75 - Actual generation per type
    A76 - Load unavailability
    A77 - Production unavailability
    A78 - Transmission unavailability
    A79 - Offshore grid infrastructure unavailability
    A80 - Generation unavailability
    A81 - Contracted reserves
    A82 - Accepted offers
    A83 - Activated balancing quantities
    A84 - Activated balancing prices
    A85 - Imbalance prices
    A86 - Imbalance volume
    A87 - Financial situation
    A88 - Cross border balancing
    A89 - Contracted reserve prices
    A90 - Interconnection network expansion
    A91 - Counter trade notice
    A92 - Congestion costs
    A93 - DC link capacity
    A94 - Non EU allocations
    A95 - Configuration document
    B11 - Flow-based allocations

Business Type:
    A29 - Already allocated capacity (AAC)
    A43 - Requested capacity (without price)
    A46 - System Operator redispatching
    A53 - Planned maintenance
    A54 - Unplanned outage
    A85 - Internal redispatch
    A95 - Frequency containment reserve
    A96 - Automatic frequency restoration reserve
    A97 - Manual frequency restoration reserve
    A98 - Replacement reserve
    B01 - Interconnector network evolution
    B02 - Interconnector network dismantling
    B03 - Counter trade
    B04 - Congestion costs
    B05 - Capacity allocated (including price)
    B07 - Auction revenue
    B08 - Total nominated capacity
    B09 - Net position
    B10 - Congestion income
    B11 - Production unit

Psr Type:
    A03 - Mixed
    A04 - Generation
    A05 - Load
    B01 - Biomass
    B02 - Fossil Brown coal/Lignite
    B03 - Fossil Coal-derived gas
    B04 - Fossil Gas
    B05 - Fossil Hard coal
    B06 - Fossil Oil
    B07 - Fossil Oil shale
    B08 - Fossil Peat
    B09 - Geothermal
    B10 - Hydro Pumped Storage
    B11 - Hydro Run-of-river and poundage
    B12 - Hydro Water Reservoir
    B13 - Marine
    B14 - Nuclear
    B15 - Other renewable
    B16 - Solar
    B17 - Waste
    B18 - Wind Offshore
    B19 - Wind Onshore
    B20 - Other
    B21 - AC Link
    B22 - DC Link
    B23 - Substation
    B24 - Transformer

Areas:
    10YSE-1--------K - Sweden
    10Y1001A1001A44P - SE1
    10Y1001A1001A45N - SE2
    10Y1001A1001A46L - SE3
    10Y1001A1001A47J - SE4
    10YPL-AREA-----S - Poland
    10YNO-0--------C - Norway
    10YNO-1--------2 - NO1
    10YNO-2--------T - NO2
    10YNO-3--------J - NO3
    10YNO-4--------9 - NO4
    10Y1001A1001A48H - NO5
    10YLV-1001A00074 - Latvia LV
    10YLT-1001A0008Q - Lithuania LT
    10Y1001A1001A83F - Germany
    10YFI-1--------U - Finland FI
    10Y1001A1001A39I - Estonia EE
    10Y1001A1001A796 - Denmark
    10YDK-1--------W - DK1
    10YDK-2--------M - DK2

"""

from prettytable import PrettyTable
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sqlite3
from pathlib import Path
from help_functions import intersection, str_to_date, cet2utc, \
    create_select_list, find_peaks, find_convex_hull, hour_min_sec
from xml.etree import ElementTree
import copy
from time import time as timefunc
from week_conversion import WeekDef

# initialize default week definition class for use in functions
weekDef = WeekDef()

reservoir_capacity = { # used to normalize reservoir values
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
}

tbidz_name = {
    'SE':'Sweden',
    'NO':'Norway',
    'DK':'Denmark',
    'FI':'Finland',
    'EE':'Estonia',
    'LT':'Lithuania',
    'LV':'Latvia',
    'GB':'Britain',
    'IE':'Ireland',
    'PT':'Portugal',
    'ES':'Spain',
    'FR':'France',
    'IT':'Italy',
    'BE':'Belgium',
    'NL':'Netherlands',
    'DE':'Germany',
    'AT':'Austria',
    'CH':'Switzerland',
    'CZ':'Czeck',
    'SK':'Slovakia',
    'SI':'Slovenia',
    'HU':'Hungary',
    'PL':'Poland',
    'CR':'Croatia',
    'SR':'Serbia',
    'BL':'Bulgaria',
    'RO':'Romania',
    'MT':'Montenegro',
    'MK':'Macedonia',
    'GR':'Greece',
    'AL':'Albania',
    'BH':'BosniaHerzergovina',
    'DK1':'DK1',
    'DK2':'DK2',
}
# bid zone keys
tbidz_key = {
    'SE1':'10Y1001A1001A44P',       
    'SE2':'10Y1001A1001A45N',
    'SE3':'10Y1001A1001A46L',
    'SE4':'10Y1001A1001A47J',
    'NO1':'10YNO-1--------2',
    'NO2':'10YNO-2--------T',
    'NO3':'10YNO-3--------J',
    'NO4':'10YNO-4--------9',
    'NO5':'10Y1001A1001A48H',
    'LV':'10YLV-1001A00074',
    'LT':'10YLT-1001A0008Q',
    'FI':'10YFI-1--------U',
    'EE':'10Y1001A1001A39I',
    'DK1':'10YDK-1--------W',
    'DK2':'10YDK-2--------M',
    'SE':'10YSE-1--------K',
    'NO':'10YNO-0--------C',
    'DK':'10Y1001A1001A796',
    'PL':'10YPL-AREA-----S',
    'NL':'10YNL----------L', # NL, TenneT
    'BE':'10YBE----------2', # Belgium
    'RU':'10Y1001A1001A49F',
    'BY':'10Y1001A1001A51S', # Belarus
    'FR':'10YFR-RTE------C',
    'DE': '10Y1001A1001A83F',  # Germany (country)
    'DE_AT_LU':'10Y1001A1001A63L', # DE-AT-LU BZ
    'DE_CZ_SK': '10YDOM-CZ-DE-SKK',  # DE-CZ-SK BZ/BZA
    'DE_LU':'10Y1001A1001A82H', # DE-LU MBA
    'DE_50HZ': '10YDE-VE-------2',  # Germany (BZA, 50 Hz CA)
    'DE_AMP':'10YDE-RWENET---I', # Germany (Amprion CA)
    'DE_TEN':'10YDE-EON------1', # Germany (TenneT CA)
    'DE_TRA':'10YDE-ENBW-----N', # Germany (TransnetBW CA)
    'IT':'10YIT-GRTN-----B', # Italy
    'IE':'10YIE-1001A00010', # Ireland
    'GB':'10YGB----------A', # Great Britain
    'PT':'10YPT-REN------W', # Portugal
    'ES':'10YES-REE------0', # Spain
    'CH':'10YCH-SWISSGRIDZ', # Switzerland
    'AT':'10YAT-APG------L', # Austria
    'CZ':'10YCZ-CEPS-----N', # Check Republic
    'SK':'10YSK-SEPS-----K', # Slovakia
    'HU':'10YHU-MAVIR----U', # Hungary
    'SI':'10YSI-ELES-----O', # Slovenia
    'CR':'10YHR-HEP------M', # Croatia
    'BL':'10YCA-BULGARIA-R', # Bulgaria
    'BH':'10YBA-JPCC-----D', # Bosnia Herzegovina
    'MK':'10YMK-MEPSO----8', # Macedonia
    'SR':'10YCS-SERBIATSOV', # Serbia
    'GR':'10YGR-HTSO-----Y', # Greece
    'RO':'10YRO-TEL------P', # Romania
    'MT':'10YCS-CG-TSO---S', # Montenegro
    'AL':'10YAL-KESH-----5', # Albania
    'UA':'10Y1001C--00003F', # Ukraine
}

# construct reverse key
tbidz_rkey = {}
for f in tbidz_key:
    tbidz_rkey[tbidz_key[f]] = f

# production type key
tpsr_rkey = {
    'B01':'Biomass',
    'B02':'Fossil Brown coal/Lignite',
    'B03':'Fossil Coal-derived gas',
    'B04':'Fossil Gas',
    'B05':'Fossil Hard coal',
    'B06':'Fossil Oil',
    'B07':'Fossil Oil shale',
    'B08':'Fossil Peat',
    'B09':'Geothermal',
    'B10':'Hydro Pumped Storage',
    'B11':'Hydro Run-of-river and poundage',
    'B12':'Hydro Water Reservoir',
    'B13':'Marine',
    'B14':'Nuclear',
    'B15':'Other renewable',
    'B16':'Solar',
    'B17':'Waste',
    'B18':'Wind Offshore',
    'B19':'Wind Onshore',
    'B20':'Other',
}

# construct reverse reverse key
tpsr_key = {}
for f in tpsr_rkey:
    tpsr_key[tpsr_rkey[f]] = f
    
# abbreviations for production types
tpsr_rabbrv = {
    'B01':'Biomass',
    'B02':'Brown coal',
    'B03':'Coal-gas',
    'B04':'Gas',
    'B05':'Hard coal',
    'B06':'Oil',
    'B07':'Oil shale',
    'B08':'Peat',
    'B09':'Geothermal',
    'B10':'Hydro pump',
    'B11':'Hydro ror',
    'B12':'Hydro res',
    'B13':'Marine',
    'B14':'Nuclear',
    'B15':'Other renew',
    'B16':'Solar',
    'B17':'Waste',
    'B18':'Wind offsh',
    'B19':'Wind onsh',
    'B20':'Other',       
        }

# construct reverse reverse key
tpsr_abbrv = {}
for f in tpsr_rabbrv:
    tpsr_abbrv[tpsr_rabbrv[f]] = f

# more broadly defined production types
aggr_types = {
        'Slow':['B01','B02','B05','B08','B17','B15','B20'], # include other renewables, other
        'Fast':['B03','B04','B06','B07'],
        'Hydro':['B10','B11','B12','B09'], # include geothermal
        'Nuclear':['B14'],
        'Wind':['B18','B19'],
        }
# solar and marine excluded
se_aggr_types = {
        'Slow':['CHP'],
        'Fast':['Gas'],
        'Hydro':['Hydro'],
        'Nuclear':['Nuclear'],
        'Wind':['Wind'],
        'Thermal':['CHP','Gas']
}
area_codes =  ['SE1','SE2','SE3','SE4','DK1','DK2','EE','LT','LV', \
             'FI', 'NO1','NO2','NO3','NO4','NO5']

# entsoe_type_map = {
#     'Thermal':['Biomass','Brown coal','Coal-gas','Gas','Hard coal','Oil','Oil shale','Peat','Waste'],
#     'Hydro':['Hydro ror','Hydro res','Hydro pump'],
#     'Wind':['Wind offsh','Wind onsh'],
#     'Nuclear':['Nuclear'],
# }
from model_definitions import entsoe_type_map

area_codes_idx = {}
for idx,c in enumerate(area_codes):
    area_codes_idx[c] = idx

country_codes = ['SE','DK','NO','FI','EE','LV','LT']    

# Production types for SE data
se_types = { 
        'Vindkraft':'Wind', # Wind onshore
        'Vattenkraft':'Hydro', # Hydro reservoir
        'Ospec':'Other', # Other
        'Solkraft':'Solar', # Solar
        'Kärnkraft':'Nuclear', # Nuclear
        'Värmekraft':'CHP', # Biomass  
        'Gas':'Gas', # Fossil gas
        }
#        prod_types = {
#                'Vindkraft':'B19', # Wind onshore
#                'Vattenkraft':'B12', # Hydro reservoir
#                'Ospec':'B20', # Other
#                'Solkraft':'B16', # Solar
#                'Kärnkraft':'B14', # Nuclear
#                'Värmekraft':'B01', # Biomass  
#                'Gas':'B04', # Fossil Gas
#        }

# periods with missing data in ENTSO-E gen per type data, will be filled with previous or following day with data
miss_gen_data_periods = [('LT',('20160203','20160209')),
                         ('LT',('20160603','20160613')),
                         ('LT',('20160326','20160326'))]

miss_load_data_periods = [('NO1',('20190208','20190211')),
                          ('NO2',('20190208','20190211')),
                          ('NO3',('20190208','20190211')),
                          ('NO4',('20190208','20190211')),
                          ('NO5',('20190208','20190211')),
                          ('LT',('20180101','20180102')),]

# fixes of individual values, in UTC time
gen_per_type_fixes = (('DK1','20191104:07','Waste',100),
                      ('DK1','20160531:09','Gas',140),
                      ('LT','20180107:06','Other',19))

# in UTC time
load_fixes = (('GB','20190126:0930',40253),
              ('GB','20190210:1430',41100),
              ('GB','20190228:1800',46300),
              ('GB','20190623:1700',33000),
              ('GB','20190623:1730',33000),
              ('GB','20190630:0930',31000),
              ('SE2','20190603:21',2000),
              ('SE2','20190603:22',2000),
              ('DK1','20190501:15',1600),
              ('DK1','20190501:16',1600)
              ) # in UTC time

reservoir_fixes = (('SE2','20161211:23',8709.5),) # in UTC time

# values below this will be replaced with nan values and interpolated
load_min_levels = {
    'GB':12000
}

class DatabaseGenUnit():
    """ Class for database with generation per unit. Due to the different structure of this database
    it is a separate class. """
    
    def __init__(self,db='C:/Data/entsoe_transparency_gen_per_unit.db'):
    
        self.db = db
        
        if Path(self.db).exists():
            pass
        

    def download_data(self,starttime='20160101',endtime='20160110',countries = ['SE','NO','DK','FI','LT','LV','EE']):
        """ Download ENTSO-E data to database. Will not overwrite any data already present
        in the database, either in TABLE units or TABLE CC_YYYYMM, thus this function can be 
        called multiple times to extend database. Time for download is approximately 40 min/month if
        all countries are included """
    
        # make sqlite database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
    
        # check if units table exists, if not create it
        c.execute("SELECT 'units' FROM sqlite_master WHERE type ='table'")
        if c.fetchone() is None:
        # create table with unit info    
            c.execute('CREATE TABLE units (' + 
                        'id TEXT NOT NULL,' + 
                        'name TEXT NOT NULL,' + 
                        'country TEXT NOT NULL,' + 
                        'area TEXT,' + 
                        'type TEXT NOT NULL,' + 
                        'resource TEXT NOT NULL' + 
                        ')')
            conn.commit()
            
        days = pd.date_range(start=str_to_date(starttime),end=str_to_date(endtime),freq='D')
            
        # get data
        ndays_dl = days.__len__()*countries.__len__()
        i = 0
        time_0 = timefunc()
        for day in days:
            if day.day == 1:
                print("------------------------")
                print("Fetching data for {0}".format(day.strftime('%Y%m')))
                print("------------------------")
            for country in countries:
                
                ## data table ##
                # name of table            
                table = country + '_' + day.strftime('%Y%m')
                
                # check if table exists, if not create table
                c.execute("SELECT name FROM sqlite_master WHERE type ='table' AND name='{0}'".format(table))
                if c.fetchone() is None:
                    # add table
                    c.execute("CREATE TABLE {0} (id TEXT NOT NULL,time TEXT NOT NULL,MWh FLOAT NOT NULL)".format(table))
                
                ## units table ##
                # get all units in table
                c.execute('SELECT id FROM units')
                unit_list = []
                for unit in c:
                    unit_list.append(unit[0])
    
                data = get_entsoe_gen_data(datatype=4,area=country,start=day.strftime('%Y%m%d'),end=day.strftime('%Y%m%d'))
                if not data is None:
                    for unit in data:
                        ## data table ##
                        for row in unit['Period'].iteritems():
                            # check if data already exists
                            c.execute("SELECT count(*) FROM {0} WHERE id = '{1}' AND time = '{2}'".format(
                                      table,unit['id'],row[0].strftime('%Y%m%d:%H')))
                            if c.fetchone()[0] == 0:
                                c.execute("INSERT INTO {0}(id,time,MWh) values ('{1}','{2}',{3})".format( 
                                          table,unit['id'],row[0].strftime('%Y%m%d:%H'),row[1]))
                        
                        ## units table ##
                        if unit['id'] not in unit_list:
                            # add unit to table
                            c.execute("INSERT INTO units(id,name,country,type,resource) values ('{0}','{1}','{2}','{3}','{4}')".format( 
                                      unit['id'],unit['name'],country,unit['production_type'],unit['registeredResource.mRID']))
                    conn.commit()
                else:
                    print("No data for {0} for {1}".format(country,day.strftime('%Y%m%d')))

                i += 1
            if day.day in [1,15]:
                runtime = timefunc() - time_0
                time_per_day = runtime / i
                days_rem = ndays_dl - i
                rem_time = days_rem*time_per_day
                rem_hr = np.floor_divide(rem_time,3600)
                rem_min = np.floor_divide(rem_time-rem_hr*3600,60)
                rem_sec = rem_time - rem_hr * 3600 - rem_min*60
                print(f'Remaining time: {rem_hr:0.0f} hr {rem_min:0.0f} min {rem_sec:0.0f} sec')
            
    def select_data(self,start='20160101',end='20160301',countries=['SE','NO','FI','DK','EE','LT','LV'],time_format='CET'):
            
        if start.__len__() <= 8:
            # no hour digits
            starttime = start + ':00'
        else:
            starttime = start
        if end.__len__() <= 8:
            endtime = end + ':23'
        else:
            endtime = end

        # convert to UTC
        if time_format == 'CET':
            starttime = cet2utc(starttime)
            endtime = cet2utc(endtime)

        
        # connect to database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
    
        # get list of all tables
        c.execute("SELECT name FROM sqlite_master WHERE type ='table'")
        all_tables = [row[0] for row in c.fetchall() if row[0] != 'units']
        
        # select those tables which are relevant for the present request
        rel_tables = [t for t in all_tables if t[:2] in countries and t[3:] >= starttime[:6] and t[3:] <= endtime[:6]]

        # create panda dataframe
        df = pd.DataFrame(index=pd.date_range(start=str_to_date(starttime),end=str_to_date((endtime)),freq='H'))
        
        # loop over tables
        for t in rel_tables:
            # get data from table
            c.execute("SELECT id,time,MWh FROM {0} WHERE time >= '{1}' AND time <= '{2}'".format(\
                      t,starttime,endtime))
            for point in c:
                df.at[str_to_date(point[1]),point[0]] = point[2]
        
        # also return generator info for all generators
        df2 = pd.DataFrame(index=df.columns,columns=['name','type','country','resource'])
        c.execute("SELECT id,name,country,type,resource FROM units")
        for point in c:
            if point[0] in df2.index:
                df2.at[point[0],'name'] = point[1]
                df2.at[point[0],'country'] = point[2]
                df2.at[point[0],'type'] = point[3]
                df2.at[point[0],'resource'] = point[4]
    
#            df.columns = df2.loc[df.columns,'name']
#            df.plot()

        # correct time index, utc-cet
        if time_format == 'CET':
            df.index = df.index + datetime.timedelta(hours=1)
        
        return df,df2



def _time2table_(timestr,table_type='flow'):
    """ Given timestr of format 'YYYYMMDD:HHMM' produce string of format 'YYYYMM'
    This function is used to map a time to a table, to be able to divide data
    into multiple tables for faster search. May be updated if further data
    division is required, e.g. one table per day
    """
    if table_type == 'flow' or table_type == 'exchange' or table_type == 'capacity':
        return timestr[0:4]  # currently one table per year
    else: # return year by default
        return timestr[0:4]

def _create_table_(c, name,table_type='flow'):
    """ Drop table if it exists, then create new table with given name """

    c.execute(f'DROP TABLE IF EXISTS {name}')

    if table_type == 'flow':
        c.execute(f'CREATE TABLE {name} (' + \
                  'time TEXT NOT NULL,' + \
                  'value REAL NOT NULL' + \
                  ')')
    elif table_type == 'gen_per_type_v2':
        c.execute(f'CREATE TABLE {name} (' + \
                  'time TEXT NOT NULL,' + \
                  'type TEXT NOT NULL,' + \
                  'value REAL NOT NULL' + \
                  ')')
    else:
        c.execute(f'CREATE TABLE {name} (' + \
                  'time TEXT NOT NULL,' + \
                  'value REAL NOT NULL' + \
                  ')')
    # print(f'_create_table_(): Unknown table_type "{table_type}"')

def _execute_(c, cmd):
    try:
        c.execute(cmd)
    except sqlite3.Error as e:
        print('Executing command ''{0}'' returned error: {1}'.format(cmd, e))

def get_tables(c):
    cmd = "SELECT name FROM sqlite_master WHERE type='table'"
    _execute_(c,cmd)
    return [t[0] for t in c.fetchall()]

def count_values(table,c):
    cmd = f"SELECT COUNT(*) FROM {table}"
    _execute_(c,cmd)
    res = c.fetchone()
    if res is not None:
        return res[0]
    else:
        return None

class Database():
    
    def __init__(self,db='Data/entsoe_transparency.db'):
        
        self.db = db
        
        if Path(self.db).exists():
            pass

    def determine_time_resolution(self,table='load_SE1'):
        # table = 'load_GB'
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        cmd = f"SELECT * FROM {table} LIMIT 5"
        cursor.execute(cmd)
        rows = []
        res = '1H'
        for row in cursor.fetchall():
            rows.append(row)
            if row[0].__len__() >= 12:
                if int(row[0][11:]) == 15:
                    res = '15min'  # quarterly resolution
                    break
        if res == '1H': # may be 30min resolution
            for row in rows:
                if row[0].__len__() >= 12:
                    if int(row[0][11:]) == 30:
                        res = '30min' # 30min resolution
                        break
        return res

    def download_cap_per_type_data(self,start_year=2016,end_year=2020,areas=None):
        """
        Download capacities per production type and store in sqlite database,
        in table cap_per_type
        """
        if areas is None:
            areas = area_codes
        # make sqlite database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
    
    
        c.execute('DROP TABLE IF EXISTS cap_per_type')
        c.execute('CREATE TABLE cap_per_type (' + \
                'year TEXT NOT NULL,' + \
                'type TEXT NOT NULL,' + \
                'area TEXT NOT NULL,' + \
                'cap REAL' + \
                ')')
        
        # download data for each year and price area
        for year in range(start_year,end_year+1):
            for area in areas:
                data = get_entsoe_gen_data(datatype=1,area=area,start='{0}0101'.format(year),end='{0}0101'.format(year))
                
                if not data is None:
                    print('Fetched data for {0} for {1}'.format(area,year))
                    # collect data
                    for point in data:
                        #acode = point['inBiddingZone_Domain.mRID']
                        # gentype = tpsr_rkey[point['MktPSRType']]
                        gentype = tpsr_rabbrv[point['MktPSRType']]
                        cmd = "INSERT INTO cap_per_type (year,type,area,cap) VALUES ('{0}','{1}','{2}',{3})".format(year,gentype,area,point['Period'][0])
                        c.execute(cmd)
                        #self.areas[tbidz_rkey[acode]].gen_cap[tpsr_rabbrv[gcode]][year] = point['Period'][0]
                        # insert data into database
                else:
                    print('Data collection failed for {0} for {1}'.format(area,year))
    
        conn.commit()
        conn.close()
        
    def select_cap_per_type_data(self,areas=None):
        """ Select data with generation capacity per type from database
        
        Output:
            df - multicolumn pandas dataframe: df[index=years,columns=[area X type]]
        """
        table = 'cap_per_type'
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
    
        cmd_min = f"SELECT min(year) FROM {table}"
        cmd_max = f"SELECT max(year) FROM {table}"
        
        c.execute(cmd_min)
        for row in c:
            startyear = row[0]
        c.execute(cmd_max)
        for row in c:
            endyear = row[0]
        if startyear is None:
            pass
        
        # create index for data frame
        dates = range(int(startyear),int(endyear)+1)
        
        # create columns
        if areas is None:
            # check which areas exist
            cmd = f'SELECT DISTINCT area FROM {table}'
            c.execute(cmd)
            area_list = []
            for row in c.fetchall():
                area_list.append(row[0])
        else:
            area_list = areas
        types = list(tpsr_key.keys())
        columns = pd.MultiIndex.from_product([area_list,types],names=['area','type'])
        df = pd.DataFrame(dtype=float,index=dates,columns=columns)
        
        # read data into df
        cmd = "SELECT year,area,type,cap FROM cap_per_type"
        if areas is not None:
            cmd += f" WHERE area in {create_select_list(areas)}"
        # print(cmd)
        c.execute(cmd)
        for row in c.fetchall():
            df.at[int(row[0]),(row[1],row[2])] = float(row[3])
        conn.close()  

        # drop nan columns
        df = df.dropna(how='all',axis=1)
        df = df.fillna(0)
        # drop zero columns
        # nzeros = (df == 0).sum()
        # df.drop(columns=[c for c in df.columns if nzeros[c] == df.__len__()],inplace=True)
        return df

    def select_cap_per_type_year(self,areas,year):
        """ Select capacity data for given year """
        table = 'cap_per_type'
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        # create columns
        if areas is None:
            # check which areas exist
            cmd = f'SELECT DISTINCT area FROM {table}'
            c.execute(cmd)
            area_list = []
            for row in c.fetchall():
                area_list.append(row[0])
        else:
            area_list = areas

        types = list(tpsr_key.keys())
        df = pd.DataFrame(dtype=float,index=area_list,columns=types)

        # read data into df
        cmd = f"SELECT area,type,cap FROM cap_per_type WHERE year in ({year})"
        if areas is not None:
            cmd += f" AND area in {create_select_list(areas)}"
        # print(cmd)
        c.execute(cmd)
        for row in c.fetchall():
            df.at[(row[0],row[1])] = float(row[2])
        conn.close()

        # drop nan columns
        df = df.dropna(how='all',axis=1)

        return df

    def download_gen_per_type_data(self,start_year = 2015,end_year=2018,areas = []):
        """ Download actual generation by production type for all bidding areas.
        The data is saved to the table "gen_per_type" in the given database:
            
        TABLE gen_per_type(TEXT time,TEXT type,TEXT area,REAL gen)
        
        time has format 'YYYYMMDD:HH'
        
        Note that some areas lacks data, such as SE1 which only has data on production
        for onshore wind. 
        """
   
        # make sqlite database
        conn = sqlite3.connect(self.db)
        #print(sqlite3.version)
        c = conn.cursor()
    
    
        c.execute('DROP TABLE IF EXISTS gen_per_type')
        c.execute('CREATE TABLE gen_per_type (' + \
                'time TEXT NOT NULL,' + \
                'type TEXT NOT NULL,' + \
                'area TEXT NOT NULL,' + \
                'gen REAL' + \
                ')')
        
        if areas == []:
            areas = ['SE1','SE2','SE3','SE4','DK1','DK2','EE','LT','LV', \
                     'FI', 'NO1','NO2','NO3','NO4','NO5']
                
        nfiles = areas.__len__() * (start_year-end_year+1) * 365
        
        print('Downloading data from entsoe transparency: ')
        for area in areas:
            # iterate over time
            date = datetime.datetime(start_year,1,1)
            counter = 0
            while date.year <= end_year:
                # retrieve data
                sdate = date.strftime('%Y%m%d')
                #print(sdate)
                # get data for one day
                data = get_entsoe_gen_data(datatype = 3,area = area,start=sdate,end=sdate,file=None)
                
                if not data is None:
                    for point in data:
                        gtype = point['production_type']
                        #area = tbidz_rkey[point['inBiddingZone_Domain.mRID']]
                        for row in point['Period'].iteritems():
                        
                            time = row[0].strftime('%Y%m%d:%H')
                            val = str(row[1]) 
                            try: # insert row into table 
                                cmd = 'INSERT INTO gen_per_type (time,type,area,gen) values("{0}","{1}","{2}",{3})'.format(time,gtype,area,val)
                                #print(cmd)
                                c.execute(cmd)
                            except sqlite3.Error as err:
                                print(err)
                                print('Area: ' + area + ', type: ' + gtype + ', time: ' + time)
                else:
                    print('Data collection failed for {0} for {1}'.format(area,sdate))
                    # increment
                date = date + datetime.timedelta(days=1)
                if np.remainder(counter,10) == 0:
                    print('Progress: {0}%'.format(str(counter/nfiles*100)[:4]))
                counter += 1
        conn.commit()
        conn.close()

    def download_gen_per_type_v2(self,startyear=2015,endyear=2019,areas=['DE'],intermediate_step=False,max_tries=10):

        # startyear = 2016
        # endyear = 2016
        tablename = 'gen_per_type_v2'
        # areas = ['DE']
        # db = Database(db='C:/Data/entsoe_transparency.db')
        #
        # data = db.select_gen_per_type_wrap(areas=['SE1','FI'])

        conn = sqlite3.connect(self.db)
        # print(sqlite3.version)
        c = conn.cursor()

        for y in range(startyear, endyear + 1):
            for a in areas:
                c.execute(f'DROP TABLE IF EXISTS {tablename}_{a}_{y}')
                c.execute(f'CREATE TABLE {tablename}_{a}_{y} (' + \
                          'time TEXT NOT NULL,' + \
                          'type TEXT NOT NULL,' + \
                          'value REAL' + \
                          ')')

        ndays_tot = 365*areas.__len__()*(endyear-startyear+1)
        i = 0
        start_time = timefunc()
        for area in areas:
            for y in range(startyear, endyear + 1):
                date = datetime.datetime(y, 1, 1)
                print(f"Downloading data for {area} for {y}")

                while date.year == y:
                    # Note: Series with 'outBiddingZone_Domain.mRID' attribute is consumption (e.g., for pumped hydro)
                    sdate = date.strftime('%Y%m%d')
                    # get data for one day
                    d = get_entsoe_gen_data(datatype=3, area=area, start=sdate, end=sdate, file=None, max_tries=max_tries)
                    # sleep(0.1)

                    if d is not None:
                        data_freq = d[0]['Period'].index.freq
                        if data_freq == '15min':
                            tformat = '%Y%m%d:%H%M'  # quarterly resolution
                        elif data_freq == '30min':
                            tformat = '%Y%m%d:%H%M'  # half hourly resolution
                        else: # hourly resolution
                            tformat = '%Y%m%d:%H'  # hourly resolution

                        if intermediate_step:
                            # intermediate step: read all values into lists
                            data_dic = {}
                            for ts in d:
                                gtype = ts['production_type']
                                if 'outBiddingZone_Domain.mRID' in ts:
                                    mult = 0 # currently: ignore consumption values
                                else:
                                    mult = 1
                                if gtype not in data_dic:
                                    data_dic[gtype] = []
                                for t,val in ts['Period'].iteritems():
                                    val_sign = val * mult
                                    # index of this hour in list (None if it has not been added)
                                    idx = next((i for i,v in enumerate(data_dic[gtype]) if v[0] == t),None)
                                    if idx is None:
                                        data_dic[gtype].append((t,val_sign))
                                    else:
                                        # add value to existing tuple
                                        data_dic[gtype].append(
                                            (t,val_sign+data_dic[gtype].pop(idx)[1])
                                        )

                            for gtype in data_dic:
                                table = f'{tablename}_{area}_{y}'
                                for t,val in data_dic[gtype]:
                                    tstr = t.strftime(tformat)
                                    cmd = f"INSERT INTO {table} (time,type,value) values('{tstr}','{gtype}',{val})"
                                    _execute_(c,cmd)

                        else:
                            for ts in d:
                                # put data directly into database (no possibility of computing net values)
                                gtype = ts['production_type']
                                table = f'{tablename}_{area}_{y}'
                                if 'outBiddingZone_Domain.mRID' in ts and gtype == 'Hydro pump':
                                    gtype = 'Hydro pump cons'
                                if 'outBiddingZone_Domain.mRID' not in ts or gtype == 'Hydro pump cons':
                                    for t,val in ts['Period'].iteritems():
                                        tstr = t.strftime(tformat)
                                        cmd = f"INSERT INTO {table} (time,type,value) values('{tstr}','{gtype}',{val})"
                                        _execute_(c,cmd)

                    date = date + datetime.timedelta(days=1)
                    i += 1
                    if date.day == 1 and date.month in [1,6]:
                        runtime = timefunc() - start_time
                        time_per_day = runtime / i
                        days_rem = ndays_tot - i
                        rem_time = days_rem*time_per_day
                        rem_hr = np.floor_divide(rem_time,3600)
                        rem_min = np.floor_divide(rem_time-rem_hr*3600,60)
                        rem_sec = rem_time - rem_hr * 3600 - rem_min*60
                        print(f'Remaining time: {rem_hr:0.0f} hr {rem_min:0.0f} min {rem_sec:0.0f} sec')

                    # break
                conn.commit()
        conn.close()

    def add_svk_gen_per_type_v2(self,data_path='D:/Data/SVK/',areas = ['SE1'],years = range(2017,2018)):
        """ Add generation per type obtained from SVK csv files """
        import csv
        import os

        data_path = Path(data_path)

        date_fmt = '%Y-%m-%d %H:%M'
        tablename = 'gen_per_type_v2'
        created_tables = []

        files = [f for f in os.listdir(data_path) if os.path.isfile(data_path / f) and '.csv' in f and 'SVK_' in f]
        rel_files = [f for f in files if f.split('_')[1] in areas and int(f.split('_')[-1].strip('.csv')) in years]

        for file in rel_files:
            #%% read csv file
            prod_type = file.split('_')[2]
            area = file.split('_')[1]
            rows = []
            with open(data_path / file,'r') as csv_file:
                csv_file = open(data_path / file,'r')
                csv_reader = csv.reader(csv_file,delimiter=';')
                ridx = 0
                for row in csv_reader:
                    if ridx > 0:
                        if row[0] == 'Summa':
                            break
                        else:
                            rows.append( (datetime.datetime.strptime(row[0],date_fmt),float(row[1].replace(',','.'))/1e3) )
                    ridx += 1

            #%% enter data into database
            conn = sqlite3.connect(self.db)
            cursor = conn.cursor()
            for row in rows:
                # note correction from CET to UTC time
                time = row[0]+datetime.timedelta(hours=-1)
                time_str = time.strftime('%Y%m%d:%H')
                year = time.year

                table = f"{tablename}_{area}_{year}"
                if table not in created_tables:
                    _create_table_(cursor,table,tablename)
                    created_tables.append(table)

                cmd = f"INSERT INTO {table} (time,type,value) values('{time_str}','{prod_type}',{row[1]})"
                _execute_(cursor,cmd)
            conn.commit()
            conn.close()

    def download_price_data(self,areas=['DE','FR','NL'],startyear=2015,endyear=2020):

        tablename = 'spotprice'

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        for a in areas:
            c.execute(f'DROP TABLE IF EXISTS {tablename}_{a}')
            c.execute(f'CREATE TABLE {tablename}_{a} (' + \
                      'time TEXT NOT NULL,' + \
                      'value REAL' + \
                      ')')

        for area in areas:
            for y in range(startyear, endyear + 1):
                print(f"Downloading data for {area} for {y}")
                for m in range(1, 13):
                    d = get_entsoe_price_data(area=area, year=y, month=m)
                    if d is not None:
                        for ts in d:
                            if ts['Period'].index.freq == '15min':
                                tformat = '%Y%m%d:%H%M'  # quarterly resolution
                            else:
                                tformat = '%Y%m%d:%H'  # hourly resolution
                            for row in ts['Period'].iteritems():
                                time = row[0].strftime(tformat)
                                year = time[:4]
                                val = str(row[1])
                                cmd = f"INSERT INTO {tablename}_{area} (time,value) values('{time}','{val}')"
                                try:
                                    c.execute(cmd)
                                except sqlite3.Error as err:
                                    print(err)
                    conn.commit()
        conn.close()

    def select_price_data(self,areas=['DE'],starttime='20180101:00',endtime='20180107:23',cet_time=False):
        """ Select data from table spotprice_AC
        """
        areas = areas.copy()
        tablename = 'spotprice'

        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        if cet_time:
            starttime = cet2utc(starttime)
            endtime = cet2utc(endtime)

        time_idx = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq='H')

        # %% find relevant tables
        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(cmd)
        # tables of right type
        rel_tables = [t[0] for t in c.fetchall() if tablename in t[0]]
        # tables for specified time range and areas
        get_tables = []
        for t in rel_tables:
            a = t.split('_')[-1]
            if a in areas:
                get_tables.append(t)

        # check if some area does not have relevant tables
        for a in areas:
            if [t for t in get_tables if a == t.split('_')[-1]] == []:
                print(f'ENTSOE Database.select_price_data: No data for {a}')
                areas.remove(a)

        # %% initialize dataframe
        data = pd.DataFrame(index=time_idx, columns=areas, dtype=float)

        # %% get data
        for t in get_tables:
            area = t.split('_')[-1]
            cmd = f"SELECT time,value FROM {t} WHERE time >= '{starttime}' AND time <= '{endtime}'"
            c.execute(cmd)
            for row in c.fetchall():
                data.at[str_to_date(row[0]), area] = row[1]

        conn.close()

        fix_pl_prices(data)

        if cet_time:
            data.index = data.index + datetime.timedelta(hours=1)
        return data

    def download_load_data(self,areas=['SE1'],startyear=2015,endyear=2016,tablename='load'):
        # areas=['SE1']
        # startyear=2015
        # endyear=2016
        # tablename = 'load'

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        for a in areas:
            c.execute(f'DROP TABLE IF EXISTS {tablename}_{a}')
            c.execute(f'CREATE TABLE {tablename}_{a} (' + \
                      'time TEXT NOT NULL,' + \
                      'value REAL' + \
                      ')')

        for area in areas:
            for y in range(startyear, endyear + 1):
                print(f"Downloading data for {area} for {y}")
                for m in range(1, 13):
                    d = get_entsoe_load_data(area=area, year=y, month=m)
                    if d is not None:
                        for ts in d:
                            if ts['Period'].index.freq in ['15min','30min']:
                                tformat = '%Y%m%d:%H%M'  # quarterly resolution
                            else:
                                tformat = '%Y%m%d:%H'  # hourly resolution
                            for row in ts['Period'].iteritems():
                                time = row[0].strftime(tformat)
                                year = time[:4]
                                val = row[1]
                                cmd = f"INSERT INTO {tablename}_{area} (time,value) values('{time}',{val})"
                                _execute_(c,cmd)
                    conn.commit()
        conn.close()

    def select_load_wrap(self,starttime='20180101:00',endtime='20180801:00',areas=['SE1','GB','DE'],cet_time=False,limit=20,
                         replace_outliers=True,print_output=True):
        """
        Note: Get load in separate series for each area, then check the time resolution, and sum time series
        which have resolution below 1 hour

        :param starttime:
        :param endtime:
        :param areas:
        :param cet_time:
        :return:
        """
        data = {
            a:self.select_load_data(starttime,endtime,areas=[a],cet_time=cet_time) for a in areas
        }
        if replace_outliers:
            for a in load_min_levels:
                if a in data:
                    df = data[a]
                    df.loc[df[a]<=load_min_levels[a],a] = np.nan

        # manually replace some outliers
        for area,tstr,val in load_fixes:
            if area in data:
                if data[area].index.freq == 'H':
                    tfmt = '%Y%m%d:%H'
                else:
                    tfmt = '%Y%m%d:%H%M'
                t = datetime.datetime.strptime(tstr,tfmt)
                if t in data[area].index:
                    data[area].at[t,area] = val

        # interpolate data before taking average
        for a in data:
            data[a].interpolate(limit=limit,inplace=True)
        nnan = sum(data[a].isna().sum().sum() for a in data)
        if nnan and print_output:
            print(f'Too many {nnan} nan values in load data, cannot interpolate values')

        df = pd.DataFrame(index=pd.date_range(start=str_to_date(starttime),end=str_to_date(endtime),freq='H'),
                          columns=areas)
        for a in areas:
            if data[a].index.freq != 'H':
                # print(f'Resampling data for {a}')
                df[a] = data[a].resample('H').mean()
            else:
                df[a] = data[a]

        for area,period in miss_load_data_periods:
            if area in df.columns:
                fill_daily_gap(df,period,area)
        return df


    def select_load_data(self,starttime='20180101:00',endtime='20180107:23',areas=['SE1'],cet_time=False):

        # areas=['SE1']
        # starttime='20180101:00'
        # endtime='20180107:23'
        # cet_time=False


        areas = areas.copy()
        tablename = 'load'



        # %% find relevant tables
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(cmd)
        # tables of right type
        rel_tables = [t[0] for t in c.fetchall() if tablename in t[0]]
        # tables for specified time range and areas
        get_tables = []
        for t in rel_tables:
            a = t.split('_')[-1]
            if a in areas:
                get_tables.append(t)

        # check if some area does not have relevant tables
        for a in areas:
            if [t for t in get_tables if a == t.split('_')[-1]] == []:
                print(f'ENTSOE Database.select_price_data: No data for {a}')
                areas.remove(a)

        #%%
        # check the frequency of the data, use table for first area
        if get_tables:
            freq = self.determine_time_resolution(get_tables[0])
        else:
            freq = 'H'
        if cet_time:
            starttime = cet2utc(f'{starttime}')
            endtime = cet2utc(f'{endtime}')

        if starttime.__len__() <= 11:
                starttime = f'{starttime}'
                endtime = f'{endtime}00'

        time_idx = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq=freq)
        # %% initialize dataframe
        data = pd.DataFrame(index=time_idx, columns=areas, dtype=float)

        # %% get data
        for t in get_tables:
            area = t.split('_')[-1]
            cmd = f"SELECT time,value FROM {t} WHERE time >= '{starttime}' AND time <= '{endtime}'"
            c.execute(cmd)
            for row in c.fetchall():
                data.at[str_to_date(row[0]), area] = row[1]

        conn.close()
        if cet_time:
            data.index = data.index + datetime.timedelta(hours=1)
        return data

    def select_gen_per_type_data(self,areas = [], types = [], starttime = '', endtime = '',excelfile = None):
        """ Select time series from sqlite database with transparency data. Data
        is returned as a pandas dataframe, and optionally exported to excel file. 
        
        Input:
            db - path to database file
            areas - list of areas to choose, by default all areas are selected
            types - list of production types, all types by default
            starttime - string with starting date in format "YYYYMMDD:HH"
            endtime - string with ending date in format "YYYYMMDD:HH"
            
        Output:
            pd_data - pandas dataframe with one column for each time series, the columns
                    are named in the manner "Area:Type", e.g. "FI:Biomass"
            
        """
        
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
            
        if areas != []:
            str_areas = '('
            for idx,area in enumerate(areas):
                if idx > 0:
                    str_areas += ",'{0}'".format(area)
                else:
                    str_areas += "'{0}'".format(area)
            str_areas += ')'
        
        if types != []:
            str_types = '('
            for idx,gtype in enumerate(types):
                if idx > 0:
                    str_types += ",'{0}'".format(gtype)
                else:
                    str_types += "'{0}'".format(gtype)
            str_types += ')'
                
        cmd = "SELECT gen,time,type,area FROM gen_per_type"
        # Note: Two additional querys are used to find the starting and ending date
        cmd_max = "SELECT max(time) FROM gen_per_type"
        cmd_min = "SELECT min(time) FROM gen_per_type" 
        
        conditions = []
        if areas!= []:
            area_cnd = 'area in ' + str_areas
            conditions.append('area')
        if types != []:
            type_cnd = 'type in ' + str_types
            conditions.append('type')
        if starttime != '':
            start_cnd = "time >= '" + starttime + "'"
            conditions.append('start')
        if endtime != '':
            end_cnd = "time <= '" + endtime + "'"
            conditions.append('end')
        
        n = conditions.__len__()
        if n > 0:
            cmd += ' WHERE ' 
            cmd_max += ' WHERE '
            cmd_min += ' WHERE '
            for idx,cnd in enumerate(conditions):
                if idx > 0:
                    cmd += ' AND '
                    cmd_max += ' AND '
                    cmd_min += ' AND '
                if cnd == 'area':
                    cmd += area_cnd
                    cmd_max += area_cnd
                    cmd_min += area_cnd
                elif cnd == 'type':
                    cmd += type_cnd
                    cmd_max += type_cnd
                    cmd_min += type_cnd
                elif cnd == 'start':
                    cmd += start_cnd
                    cmd_max += start_cnd
                    cmd_min += start_cnd
                elif cnd == 'end':
                    cmd += end_cnd
                    cmd_max += end_cnd
                    cmd_min += end_cnd
                else:
                    print('Unknown condition type: {0}'.format(c))
                    
        #print(cmd_min)
        c.execute(cmd_min)
        for row in c:
            start = row[0]
        c.execute(cmd_max)
        for row in c:
            end = row[0]
        if start is None:
            print('The following command returned no data: {0}'.format(cmd))
            return None
            
        # create index for data frame
        sdate = datetime.datetime(int(start[0:4]),int(start[4:6]),int(start[6:8]),int(start[9:11]))
        edate = datetime.datetime(int(end[0:4]),int(end[4:6]),int(end[6:8]),int(end[9:11]))
        
        dates = pd.date_range(start=sdate,end=edate,freq='H')
        
        # find columns for data frame
        if areas == []: # all areas selected by default
            areas = area_codes
        if types == []:
            types = list(tpsr_abbrv.keys())
        # create header for each combination of area and type
    #    cols = []
    #    for area in areas:
    #        for gtype in types: 
    #            cols.append(area + ':' + gtype)
        
        # allocate panda data frame for each area
        gdata = {}
        for area in areas:
            gdata[area] = pd.DataFrame( \
                    dtype = float, \
                    index=dates, \
                    columns=types)
            
    #    # allocate panda data frame for data    
    #    pd_data = pd.DataFrame( \
    #                dtype = float, \
    #                index=dates, \
    #                columns=cols)
        
        # get data
        c.execute(cmd) # SELECT gen,time,type,area FROM gen_per_type
        for row in c:
            date = datetime.datetime(int(row[1][0:4]),int(row[1][4:6]),int(row[1][6:8]),int(row[1][9:11]))
            #pd_data[row[3] + ':' + row[2]][date] = row[0]
            gdata[row[3]][row[2]][date] = row[0]
        
        conn.close()    
    
    
        # remove all columns which are NaN
        #    isnan = pd_data.isnull().sum()
        #    dropcols = []
        #    for row in isnan.iteritems():
        #        if row[1] == pd_data.__len__():
        #            dropcols.append(row[0])     
        #    pd_data = pd_data.drop(columns=dropcols)
        
        for area in areas:
            isnan = gdata[area].isnull().sum()
            dropcols = []
            for row in isnan.iteritems():
                if row[1] == gdata[area].__len__():
                    dropcols.append(row[0])     
            gdata[area] = gdata[area].drop(columns=dropcols)
    
        if not excelfile is None:
            writer = pd.ExcelWriter(excelfile)
            for area in areas:
                gdata[area].to_excel(writer,sheet_name=area)
            writer.save()
            #pd_data.to_excel(excelfile)

        return gdata

    def select_gen_per_type_v2(self,starttime='20170101:00',endtime='20170108:00',areas=['DE'],types=None,cet_time=False):
        """
        :param starttime:
        :param endtime:
        :param areas:
        :param types:
        :param time_format: UTC/ECT
        :return:
        """
        areas = areas.copy()
        tablename = 'gen_per_type_v2'
        # areas = ['DE']
        # types = None
        # types = ['Biomass', 'Brown coal']
        # starttime = '20170101:00'
        # endtime = '20170107:23'

        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        if types is not None:
            str_types = create_select_list(types)

        if cet_time:
            starttime = cet2utc(starttime)
            endtime = cet2utc(endtime)
        # years
        startyear = int(starttime[:4])
        endyear = int(endtime[:4])

        time_idx_hour = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq='H')
        time_idx_quarter = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq='15min')
        time_idx_halfhour = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq='30min')

        # %% find relevant tables
        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(cmd)
        # tables of right type
        rel_tables = [t[0] for t in c.fetchall() if tablename in t[0]]
        # tables for specified time range and areas
        get_tables = []
        for t in rel_tables:
            y = int(t.split('_')[-1])
            a = t.split('_')[-2]
            if y >= startyear and y <= endyear and a in areas:
                get_tables.append(t)
        # check if some area does not have relevant tables
        get_areas = []
        for a in areas:
            if [t for t in get_tables if a == t.split('_')[-2]] == []:
                print(f'ENTSOE Database.select_data_v2: No data for {a}')
            else:
                get_areas.append(a)
        areas = get_areas
        # %% find types for each area, for dataframe initialization
        area_types = {}
        for a in areas:
            area_types[a] = []
        for t in get_tables:
            a = t.split('_')[-2]
            cmd = f"SELECT DISTINCT type FROM {t}"
            c.execute(cmd)
            for col in c.fetchall():
                if types is None:
                    if col[0] not in area_types[a]:
                        area_types[a].append(col[0])
                else:
                    # print(col)
                    if col[0] not in area_types[a] and col[0] in types:
                        area_types[a].append(col[0])

        # %% find resolution for each area
        area_res = {}
        for a in areas:
            area_res[a] = 'H'
            table = [t for t in get_tables if a == t.split('_')[-2]][0]
            cmd = f"SELECT * FROM {table} LIMIT 5"
            c.execute(cmd)
            rows = []
            for row in c.fetchall():
                rows.append(row)
                if row[0].__len__() >= 12:
                    if int(row[0][11:]) == 15:
                        area_res[a] = 'Q'  # quarterly resolution
                        break
            for row in rows:
                if row[0].__len__() >= 12:
                    if int(row[0][11:]) == 30:
                        area_res[a] = 'T'
                        break

        # %% initialize dataframes
        data = {}
        for a in areas:
            if area_res[a] == 'H':
                data[a] = pd.DataFrame(index=time_idx_hour, columns=area_types[a], dtype=float)
            elif area_res[a] == 'Q':
                data[a] = pd.DataFrame(index=time_idx_quarter, columns=area_types[a], dtype=float)
            elif area_res[a] == 'T':
                data[a] = pd.DataFrame(index=time_idx_halfhour, columns=area_types[a], dtype=float)

        # %% get data
        for t in get_tables:
            a = t.split('_')[-2]
            if area_res[a] == 'Q' and starttime.__len__() < 13:
                # add zeros in case it was forgotten for time format :DDMM
                cmd = f"SELECT time,type,value FROM {t} WHERE time >= '{starttime}00' AND time <= '{endtime}00'"
            else:
                cmd = f"SELECT time,type,value FROM {t} WHERE time >= '{starttime}' AND time <= '{endtime}'"
            if types is not None:
                cmd += f" AND type IN {str_types}"

            c.execute(cmd)
            for row in c.fetchall():
                data[a].at[str_to_date(row[0]), row[1]] = row[2]

        conn.close()
        if cet_time:
            for a in data:
                data[a].index = data[a].index + datetime.timedelta(hours=1)
        return data

    def select_gen_per_type_wrap_v2(self,starttime='20190101:00',endtime='20190107:23',
                                    areas=['SE1','SE2','FI'],cet_time=False,
                                    type_map=entsoe_type_map,dstfix=True,drop_data=True,limit=20,print_output=True,drop_pc=None,limit_edge=4):
        """
        Select generation data using select_gen_per_type_v2 and aggregate categories

        Note:
        - faster than select_gen_per_type, also allows option for CET/UTC time
        - does not impute missing NO data for 2017
        :return:
        """
        #%%
        # starttime = '20190101:00'
        # endtime = '20190107:23'

        # Aggregate types to be returned
        gen_types = ['Hydro','Thermal','Nuclear','Wind','Solar']
        excl_types = ['Hydro pump cons'] # types to exclude when aggregating total production

        data_raw = self.select_gen_per_type_v2(areas=areas,starttime=starttime,endtime=endtime,cet_time=cet_time)

        if drop_pc is not None:
            # drop columns with more than "drop_pc" missing data
            for a in data_raw:
                df = data_raw[a]
                dcols = [c for c in df.columns if df[c].isna().sum()/df.__len__() >= drop_pc/100]
                if dcols:
                    if print_output:
                        print(f'Dropping columns for {a}: {dcols}')
                    df.drop(columns=dcols,inplace=True)

        data_raw = fix_gen_data_manual(data_raw,cet_time)
        # replace missing zero data with nan values
        fix_zero_perios(data_raw,print_output=print_output)

        # interpolate nan values
        for i,df in data_raw.items():
            df.interpolate(limit=limit, inplace=True)  # interpolate up to limit samples
            # fill edges of data
            df.fillna(method='ffill',inplace=True,limit=limit_edge)
            df.fillna(method='bfill',inplace=True,limit=limit_edge)
            if print_output:
                for gtype in df.columns:
                    nnan = df[gtype].isna().sum()
                    if nnan > 0:
                        print(f'Too many {nnan} nan values for {i} {gtype} to interpolate')

        # create hourly data
        data = {}
        for a in data_raw:
            if data_raw[a].index.freq == '15min' or data_raw[a].index.freq == '30min':
                # get hourly data
                data[a] = data_raw[a].resample('H').mean()
            else:
                data[a] = data_raw[a]

        if dstfix: # Fix ENTSO-E data: missing values for CEST/CET shift
            data = fix_time_shift(data,cet_time=cet_time,print_output=print_output)

        # aggregate total production
        for a in data:
            df = data[a]
            df['Tot'] = df.loc[:,[c for c in df.columns if c not in excl_types]].sum(axis=1)

        gen_in_area = {}
        gen_in_area_exist = {} # True if type exists in data already, if not it needs to be computed
        aggr_types = {}
        for a in areas:
            gen_in_area[a] = []
            gen_in_area_exist[a] = []
            aggr_types[a] = []
            for g in gen_types:
                if g in data[a].columns:
                    gen_in_area[a].append(g)
                    gen_in_area_exist[a].append(True)
                elif g in type_map:
                    for sub_type in type_map[g]:
                        # check if any relevant columns for computing generation type exists
                        if sub_type in data[a].columns:
                            gen_in_area[a].append(g)
                            gen_in_area_exist[a].append(False)
                            aggr_types[a].append(g)
                            break

        for a in areas:

            # compute aggregate types
            for g in aggr_types[a]:
                data[a][g] = data[a].loc[:,[s for s in type_map[g] if s in data[a].columns]].sum(axis=1,skipna=True)
            # drop columns
            if drop_data:
                data[a].drop(labels=[c for c in data[a].columns if c not in gen_in_area[a]+['Tot']],axis=1,inplace=True)

        return data

    def get_se_gen_data(self):
        """
        Enter SvK generation data per type into separate table:
            
            se_gen_per_type(TEXT time, TEXT hype, TEXT area, FLOAT gen)
        
        Data comes from excel files from SvK homepage
        
        """
    
        # get SvK production data
        
        data_path = 'C:/Users/elisn/Box Sync/Data/SvK/'

        import xlrd
        
        # connect to sqlite database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        
        # create separeate table for SE data
        c.execute('DROP TABLE IF EXISTS se_gen_per_type')
        c.execute('CREATE TABLE se_gen_per_type (' + \
                    'time TEXT NOT NULL,' + \
                    'type TEXT NOT NULL,' + \
                    'area TEXT NOT NULL,' + \
                    'gen REAL' + \
                    ')')
        
        files = [
                'statistik-per-elomrade-och-timme-2018.xls',
                'timvarden-2017-01-12.xls',
                'timvarden-2016-01-12.xls',
                'statistik-per-timme-och-elomrade-2015.xls',
                'timvarden-2014-01-12.xls',
                'timvarden-2013-01-12.xls',
                'timvarden-2012-01-12.xls',
                'timvarden-2011-01-12.xls',
        ]
        
        #file_name = 'statistik-per-elomrade-och-timme-2018.xls'
        
        for file_name in files:
        
            print('Reading {0}'.format(file_name))
            wb = xlrd.open_workbook(data_path + file_name)
            ws = wb.sheet_by_index(0)
            
            headers1 = ws.row_values(0)
            headers2 = ws.row_values(1)
            headers = [x[0]+x[1] for  x in zip(headers1,headers2)]
            
            areas = ws.row_values(2)
            
            # check which columns to keep
            col_idxs = []
            col_names = []
            for idx,col in enumerate(headers):
                if 'Vindkraft' in col:
                    col_idxs.append(idx)
                    col_names.append('Vindkraft')
                elif 'Vattenkraft' in col:
                    col_idxs.append(idx)
                    col_names.append('Vattenkraft')
                elif 'Ospec' in col:
                    col_idxs.append(idx)
                    col_names.append('Ospec')
                elif 'Solkraft' in col:
                    col_idxs.append(idx)
                    col_names.append('Solkraft')
                elif 'Kärnkraft' in col:
                    col_idxs.append(idx)
                    col_names.append('Kärnkraft')
                elif 'Värmekraft' in col:
                    col_idxs.append(idx)
                    col_names.append('Värmekraft')
                elif 'Gast' in col:
                    col_idxs.append(idx)
                    col_names.append('Gas')
                   
            
            ridx = 0
            for row in ws.get_rows():
                
                if ridx >= 5:
                    # get current datetime
                    if type(row[0].value) is str:
                        timeinfo = row[0].value.replace('.',' ').split(' ')
                        day = timeinfo[0]
                        if day.__len__() < 2:
                            day = '0' + day
                        month = timeinfo[1]
                        if month.__len__() < 2:
                            month = '0' + month
                        hour = timeinfo[3]
                        hour = hour.split(':')[0]
                        if hour.__len__() < 2:
                            hour = '0' + hour
                        timestr = timeinfo[2]+month+day+':'+hour
                    elif type(row[0].value) is float:
                        py_date = xlrd.xldate.xldate_as_datetime(row[0].value,wb.datemode)
                        timestr = py_date.strftime('%Y%m%d:%H')
                    for nidx,cidx in enumerate(col_idxs):
                        area = areas[cidx]
                        #gtype = tpsr_rabbrv[prod_types[col_names[nidx]]]
                        gtype = se_types[col_names[nidx]]
                        data = row[cidx].value
                        if not type(data) is float:
                            data = 'NULL'
                        cmd = "INSERT INTO se_gen_per_type (time,type,area,gen) VALUES ('{0}','{1}','{2}',{3})".format(timestr,gtype,area,data) 
                        #print(cmd)
                        c.execute(cmd)
                ridx += 1
        
        conn.commit()
        conn.close()
        
    def select_se_gen_per_type_data(self,areas = [], types = [], starttime = '', endtime = '',excelfile = None):
        """ Select production per type data from SE table 
        
        Input:
            db - path to database file
            areas - list of areas to choose, by default all areas are selected
            types - list of production types, all types by default
            starttime - string with starting date in format "YYYYMMDD:HH"
            endtime - string with ending date in format "YYYYMMDD:HH"
            
        Output:
            pd_data - pandas dataframe with one column for each time series, the columns
                    are named in the manner "Area:Type", e.g. "FI:Biomass"
            
        """
        
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
            
        if areas != []:
            str_areas = '('
            for idx,area in enumerate(areas):
                if idx > 0:
                    str_areas += ",'{0}'".format(area)
                else:
                    str_areas += "'{0}'".format(area)
            str_areas += ')'
        
        if types != []:
            str_types = '('
            for idx,gtype in enumerate(types):
                if idx > 0:
                    str_types += ",'{0}'".format(gtype)
                else:
                    str_types += "'{0}'".format(gtype)
            str_types += ')'
                
        cmd = "SELECT gen,time,type,area FROM se_gen_per_type"
        # Note: Two additional querys are used to find the starting and ending date
        cmd_max = "SELECT max(time) FROM se_gen_per_type"
        cmd_min = "SELECT min(time) FROM se_gen_per_type" 
        
        conditions = []
        if areas!= []:
            area_cnd = 'area in ' + str_areas
            conditions.append('area')
        if types != []:
            type_cnd = 'type in ' + str_types
            conditions.append('type')
        if starttime != '':
            start_cnd = "time >= '" + starttime + "'"
            conditions.append('start')
        if endtime != '':
            end_cnd = "time <= '" + endtime + "'"
            conditions.append('end')
        
        n = conditions.__len__()
        if n > 0:
            cmd += ' WHERE ' 
            cmd_max += ' WHERE '
            cmd_min += ' WHERE '
            for idx,cnd in enumerate(conditions):
                if idx > 0:
                    cmd += ' AND '
                    cmd_max += ' AND '
                    cmd_min += ' AND '
                if cnd == 'area':
                    cmd += area_cnd
                    cmd_max += area_cnd
                    cmd_min += area_cnd
                elif cnd == 'type':
                    cmd += type_cnd
                    cmd_max += type_cnd
                    cmd_min += type_cnd
                elif cnd == 'start':
                    cmd += start_cnd
                    cmd_max += start_cnd
                    cmd_min += start_cnd
                elif cnd == 'end':
                    cmd += end_cnd
                    cmd_max += end_cnd
                    cmd_min += end_cnd
                else:
                    print('Unknown condition type: {0}'.format(c))
                    
        #print(cmd_min)
        c.execute(cmd_min)
        for row in c:
            start = row[0]
        c.execute(cmd_max)
        for row in c:
            end = row[0]
        if start is None:
            print('The following command returned no data: {0}'.format(cmd))
            return None
            
        # create index for data frame
        sdate = datetime.datetime(int(start[0:4]),int(start[4:6]),int(start[6:8]),int(start[9:11]))
        edate = datetime.datetime(int(end[0:4]),int(end[4:6]),int(end[6:8]),int(end[9:11]))
        
        dates = pd.date_range(start=sdate,end=edate,freq='H')
        
        # find columns for data frame
        if areas == []: # all areas selected by default
            areas = ['SE1','SE2','SE3','SE4']
        if types == []:
            types = [se_types[f] for f in se_types]

        
        # allocate panda data frame for each area
        gdata = {}
        for area in areas:
            gdata[area] = pd.DataFrame( \
                    dtype = float, \
                    index=dates, \
                    columns=types)
        
        # get data
        c.execute(cmd) # SELECT gen,time,type,area FROM gen_per_type
        for row in c:
            date = datetime.datetime(int(row[1][0:4]),int(row[1][4:6]),int(row[1][6:8]),int(row[1][9:11]))
            gdata[row[3]][row[2]][date] = row[0]
        
        conn.close()    
    
        for area in areas:
            isnan = gdata[area].isnull().sum()
            dropcols = []
            for row in isnan.iteritems():
                if row[1] == gdata[area].__len__():
                    dropcols.append(row[0])     
            gdata[area] = gdata[area].drop(columns=dropcols)
    
        if not excelfile is None:
            writer = pd.ExcelWriter(excelfile)
            for area in areas:
                gdata[area].to_excel(writer,sheet_name=area)
            writer.save()

        return gdata

    def get_entsoe_production_stats(self,starttime='20180101:00',endtime='20181231:23',thermal=['Biomass','Brown coal','Coal-gas','Gas','Hard coal','Oil','Oil shale','Peat','Waste'],excelfile=None):
        """ Find min, max and mean values for the different production types
        based on entso-e transparency data, for generation according to different
        categories
        Also find maximum ramping rates
        """
        
        # get entso-e actual production data
        #pd_data = self.select_gen_per_type_data(areas=area_codes,starttime=starttime,endtime=endtime,excelfile=excelfile)
       
        # aggregate production over broader types
#        adata = {}
#        for area in pd_data.keys():
#            adata[area] = aggregate_gen_per_type_data(pd_data[area])
#            adata[area]['Total'] = adata[area].sum(axis=1)
#            adata[area]['Thermal'] = adata[area].loc[:,[f for f in ['Fast','Slow'] if f in adata[area].columns]].sum(axis=1)
#            
#        
#        # get se production data
#        se_data = self.select_se_gen_per_type_data(starttime=starttime,endtime=endtime)
#        for area in se_data.keys():
#            adata[area] = pd.DataFrame(columns=list(se_aggr_types.keys()),index=se_data[area].index)
#            for gtype in se_aggr_types.keys():
#                # check if any column exist in se-data
#                cols = [c for c in se_data[area].columns if c in se_aggr_types[gtype]]
#                if cols != []:
#                    adata[area].loc[:,gtype] = se_data[area].loc[:,cols].sum(axis=1)
#                
        pd_data = self.select_gen_per_type_wrap(starttime=starttime,endtime=endtime,thermal=thermal)
        adata = {}
        for area in pd_data:
            adata[area] = pd_data[area].drop(columns=[col for col in pd_data[area].columns if col not in ['Thermal','Hydro','Nuclear','Wind']])
        # for each time series, determine min, max, and average
        stats = {}
        for area in adata.keys():
            astats = {}
            # first replace zero values with NaN
            adata[area][adata[area]==0] = np.nan
            for icol,col in enumerate(adata[area].columns):
                cstats = {}
                cstats['min'] = np.min(adata[area][col])
                cstats['max'] = np.max(adata[area][col])
                cstats['avg'] = np.mean(adata[area][col])
                diff = adata[area][col].diff()
                cstats['maxramp'] = np.max(diff)
                cstats['minramp'] = np.min(diff)
                astats[col]=cstats
                
            stats[area]=astats
        
        return stats
    
    def select_gen_per_type_wrap(self,starttime='20180101:00',endtime='20180107:23',areas=area_codes,dstfix=True,thermal=['Biomass','Brown coal','Coal-gas','Gas','Hard coal','Oil','Oil shale','Peat','Waste'],hydro=['Hydro ror','Hydro res','Hydro pump'],wind=['Wind offsh','Wind onsh']):
        """ 
        Wrapper for select functions. Selects data from ENTSO-E for non-SE regions, and from
        SvK for SE regions. ENTSO-E data is also time-displaced one hour to fix time lag. 
        Also the aggregate production for categories 'Hydro' and 'Thermal' are computed
        according to the given definitions. 
        
        Thermal: ['Biomass',Brown coal','Coal-gas','Gas','Hard coal','Oil','Oil shale','Peat','Waste']
        'B01':'Biomass',
        'B02':'Brown coal',
        'B03':'Coal-gas',
        'B04':'Gas',
        'B05':'Hard coal',
        'B06':'Oil',
        'B07':'Oil shale',
        'B08':'Peat',
        Waste
    """
        se_areas = ['SE1','SE2','SE3','SE4']
        # check if all price areas are in SE
        only_SE = True
        has_SE = False
        has_NO = False
        for a in areas:
            if a not in se_areas:
                only_SE = False
            else:
                has_SE = True
            if a in [f'NO{i}' for i in range(1,6)]:
                has_NO = True


        # get entsoe data
        if not only_SE:
            pd_data = self.select_gen_per_type_data(areas=[a for a in areas if 'SE' not in a], \
                        starttime=(str_to_date(starttime)+datetime.timedelta(hours=-1)).strftime('%Y%m%d:%H'), \
                        endtime=(str_to_date(endtime)+datetime.timedelta(hours=-1)).strftime('%Y%m%d:%H'))

            utc_start = (str_to_date(starttime)+datetime.timedelta(hours=-1)).strftime('%Y%m%d:%H')
            utc_end = (str_to_date(endtime)+datetime.timedelta(hours=-1)).strftime('%Y%m%d:%H')

            if has_NO:
                # Fix for missing NO data, September 17-19 2017
                mstart = '20170916:22'
                mend = '20170919:23'
                dmstart = str_to_date(mstart)
                dmend = str_to_date(mend)
                # check if any dates with missing data fall inside interval
                if (utc_start <= mstart and utc_end >= mstart) or (utc_start <= mend and utc_end >= mend):

                    # get data for previous day
                    impute_data = self.select_gen_per_type_data(starttime='20170915:00',endtime='20170915:23',areas=[f"NO{i}" for i in range(1,6)])
                    entsoe_idx = pd_data[list(pd_data)[0]].index
                    N = entsoe_idx.__len__()
                    tidx = 0
                    while tidx < N:
                        tidx += 1
                        if entsoe_idx[tidx] >= dmstart:
                            sidx = tidx
                            break

                    for area in [f"NO{i}" for i in range(1,6)]:
                        if area in pd_data:
                            # replace values with those from previous day
                            tidx = sidx
                            count = 0
                            while tidx < N:
                                if entsoe_idx[tidx] > dmend:
                                    break
                                # check if value inside period
                                if entsoe_idx[tidx] >= dmstart and entsoe_idx[tidx] <= dmend:
                                    miss_cols = pd_data[area].iloc[tidx,:].isna()
                                    pd_data[area].loc[entsoe_idx[tidx],miss_cols] = np.array(impute_data[area].loc[impute_data[area].index[entsoe_idx[tidx].hour],:])
                                    count += miss_cols.sum()
                                tidx += 1
                            print(f"Imputed {count} values for {area} for period {mstart}-{mend}")

            # correct index for ENTSO-E data: UTC -> UTC + 1
            for area in pd_data:
                new_index = [t + datetime.timedelta(hours=1) for t in pd_data[area].index]
                pd_data[area].index = new_index

            # aggregate hydro and thermal generation
            for area in pd_data:
                pd_data[area].loc[:,'Hydro'] = pd_data[area].loc[:,[h for h in hydro if h in pd_data[area].columns]].sum(axis=1,skipna=False)
                pd_data[area].loc[:,'Thermal'] = pd_data[area].loc[:,[h for h in thermal if h in pd_data[area].columns]].sum(axis=1,skipna=False)
                pd_data[area].loc[:,'Wind'] = pd_data[area].loc[:,[h for h in wind if h in pd_data[area].columns]].sum(axis=1,skipna=False)

        #%% get SE data
        if has_SE:
            pd_data_se = self.select_se_gen_per_type_data(areas=[a for a in areas if a in se_areas],starttime=starttime, endtime=endtime)

            if not only_SE: # has entso-e data: copy SE data into existing dataframe
                for area in [a for a in areas if 'SE' in a]:
                    pd_data[area] = pd.DataFrame(index=pd_data_se[area].index,columns=['Hydro','Thermal','Solar'])
                    pd_data[area].loc[:,'Hydro'] = pd_data_se[area].loc[:,'Hydro']
                    pd_data[area].loc[:,'Thermal'] = pd_data_se[area].loc[:,'CHP']

                    if 'Nuclear' in pd_data_se[area].columns:
                        pd_data[area].loc[:,'Nuclear'] = pd_data_se[area].loc[:,'Nuclear']
                    if 'Wind' in pd_data_se[area].columns:
                        pd_data[area].loc[:,'Wind'] = pd_data_se[area].loc[:,'Wind']
                    if 'Solar' in pd_data_se[area].columns:
                        pd_data[area].loc[:,'Solar'] = pd_data_se[area].loc[:,'Solar']
            else: # no entso-e data, rename columns of se dataframe
                for a in pd_data_se:
                    cols = []
                    for c in pd_data_se[a].columns:
                        if c == 'CHP':
                            cols.append('Thermal')
                        else:
                            cols.append(c)
                    pd_data_se[a].columns = cols
                pd_data = pd_data_se

        if dstfix and not only_SE: # Fix ENTSO-E data: missing values for CEST/CET shift
            pd_data = fix_entsoe(pd_data)

        return pd_data
        

    def drop_tables(self):
        """ Drop all tables """
            # make sqlite database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
    
        # drop all tables
        c.execute("SELECT name FROM sqlite_master WHERE type ='table'")
        for tab in c.fetchall():
            c.execute("DROP TABLE '{0}'".format(tab[0]))

        conn.commit()


    def download_forecast_data(self,startyear=2014,endyear=2019,areas=['SE1','SE2','SE3','SE4','DK1','DK2','FI','LV','LT','EE','NO1','NO2','NO3','NO4','NO5']):
        """ Download wind forecast (onshore and offshore) data
            May be extended to solar forecast data
        """

        conn = sqlite3.connect(self.db)
        # print(sqlite3.version)
        c = conn.cursor()

        for y in range(startyear, endyear + 1):
            c.execute(f'DROP TABLE IF EXISTS day_ahead_forecast_{y}')
            c.execute(f'CREATE TABLE day_ahead_forecast_{y} (' + \
                      'time TEXT NOT NULL,' + \
                      'type TEXT NOT NULL,' + \
                      'area TEXT NOT NULL,' + \
                      'value REAL' + \
                      ')')

        for area in areas:
            for y in range(startyear, endyear + 1):
                print(f"Downloading data for {area} for {y}")
                starttime = f"{y}0101"
                endtime = f"{y}1231"

                d = get_entsoe_gen_data(datatype=5, area=area, start=starttime, end=endtime)
                if not d is None:
                    for ts in d:
                        gtype = ts['production_type']
                        for row in ts['Period'].iteritems():
                            time = row[0].strftime('%Y%m%d:%H')
                            year = time[:4]
                            val = str(row[1])
                            cmd = f"INSERT INTO day_ahead_forecast_{year} (time,type,area,value) values('{time}','{gtype}','{area}','{val}')"
                            try:
                                c.execute(cmd)
                            except sqlite3.Error as err:
                                print(err)
                                print('Area: ' + area + ', type: ' + gtype + ', time: ' + time)

            conn.commit()
        conn.close()

    def select_data(self,table='day_ahead_forecast',areas=[],categories=[],starttime='20180101:00',endtime='20180107:23',time_format='CET',fix_time_range=False):
        """
        Generic method for data selction. Works without major modifications on all tables with the
        columns: (time, type, area, data). Works for both tables where data is split by years
        and for tables with all data in one table. The first option is prefferred for data storage
        since it is significantly faster for large amounts of data.

        Tables implemented so far:
        day_ahead_forecast_XXXX (data=value)
        gen_per_type (data=gen)

        :param table: table name
        :param areas: list with selected areas (data for all areas is returned if areas = [])
        :param categories: list with selected categories (defaults to all categories)
        :param starttime: starting time, 'YYYYMMDD:HH'
        :param endtime: ending time, 'YYYYMMDD:HH'
        :param time_format: time zone for speficied times, 'CET'/'UTC', as data is stored in UTC for most tables
        :param fix_time_range: if true returned data frames will have full time range, even if data is missing
        :return: d - dict with one dataframe per area
        """
        conn = sqlite3.connect(self.db)
        # print(sqlite3.version)
        c = conn.cursor()

        # %% Select day ahead data from tables
        # time_format = 'CET'
        # starttime = '20161201:00'
        # endtime = '20180112:00'
        # categories = []
        # areas = []
        # categories = ['Wind onsh', 'Wind offsh']
        # areas = ['SE4', 'DK1']
        # table = 'gen_per_type'
        # fix_time_range = False

        if table == 'gen_per_type':
            datacol = 'gen'
        elif table == 'day_ahead_forecast':
            datacol = 'value'
        else:
            pass
            print(f'Unknown table {table}')
            return None

        # convert to UTC
        if time_format == 'CET':
            utc_start = cet2utc(starttime)
            utc_end = cet2utc(endtime)

        yr1 = utc_start[:4]
        yr2 = utc_end[:4]

        start_cnd = 'time >= ' + "'{0}'".format(utc_start)
        end_cnd = 'time <= ' + "'{0}'".format(utc_end)

        # find relevant tables
        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(cmd)
        rel_tables = [t[0] for t in c.fetchall() if table in t[0]]

        # check if tables are separated by years (gives faster data selection)
        try:
            int(rel_tables[0].split('_')[-1])
            yearly_tables = True
        except:
            yearly_tables = False
        # %%
        # find tables for requested time range
        if yearly_tables:
            rel_tables = [t for t in rel_tables if t.split('_')[-1] >= utc_start[:4] and t.split('_')[-1] <= utc_end]
        else:
            # only look in table giving exact match for name
            rel_tables = [table]

        conditions = []
        if categories != []:
            cat_cnd = 'type in ' + create_select_list(categories)
            conditions.append(cat_cnd)
        if areas != []:
            area_cnd = 'area in ' + create_select_list(areas)
            conditions.append(area_cnd)

        # find min and max times
        # Note: if data is missing for some timerange
        mintimes = []
        maxtimes = []
        for tn in rel_tables:
            cmd_min = f"SELECT min(time) FROM {tn}"
            cmd_max = f"SELECT max(time) FROM {tn}"
            for idx, cnd in enumerate(conditions):
                if idx == 0:
                    cmd_min += f" WHERE {cnd}"
                    cmd_max += f" WHERE {cnd}"
                else:
                    cmd_min += f" AND {cnd}"
                    cmd_max += f" AND {cnd}"
            if yearly_tables:
                yr = tn.split('_')[-1]
                if yr == yr1 and yr == yr2:
                    cmd_min += f" AND {start_cnd} AND {end_cnd}"
                    cmd_max += f" AND {start_cnd} AND {end_cnd}"
                elif yr == yr1:
                    cmd_min += f" AND {start_cnd}"
                    cmd_max += f" AND {start_cnd}"
                elif yr == yr2:
                    cmd_min += f" AND {end_cnd}"
                    cmd_max += f" AND {end_cnd}"
            else:
                cmd_min += f" AND {start_cnd} AND {end_cnd}"
                cmd_max += f" AND {start_cnd} AND {end_cnd}"
            c.execute(cmd_min)
            vals = c.fetchall()
            if not vals[0][0] is None:
                mintimes.append(vals[0][0])
            c.execute(cmd_max)
            vals = c.fetchall()
            if not vals[0][0] is None:
                maxtimes.append(vals[0][0])
        if mintimes.__len__() == 0:
            print(f"Found no values satisfying criteria in {table}")
            for cnd in conditions:
                print(cnd)
            return None

        mintime = min(mintimes)
        maxtime = max(maxtimes)

        if categories == []:
            # get types from database
            columns = []
            for tn in rel_tables:
                cmd = "SELECT DISTINCT type FROM {0}".format(tn)
                c.execute(cmd)
                columns = list(set([val[0] for val in c.fetchall()] + columns))
        else:
            columns = categories

        if areas == []:
            # get areas from database
            alist = []
            for tn in rel_tables:
                cmd = "SELECT DISTINCT area FROM {0}".format(tn)
                c.execute(cmd)
                alist = list(set([val[0] for val in c.fetchall()] + alist))
            areas = alist
        else:
            areas = areas

        # initialize data storage,
        # {area:DataFrame(index=times,columns=types)
        d = {}
        for a in areas:
            if fix_time_range:
                # create data frame for whole specified time range, irrespective of whether
                # data is missing in the database
                d[a] = pd.DataFrame(data=np.nan, dtype=float, columns=columns, \
                                    index=pd.date_range(start=str_to_date(utc_start), end=str_to_date(utc_end),
                                                        freq='H'))

            else:
                # use dynamic data range, i.e. start time index for the first non-missing value
                d[a] = pd.DataFrame(data=np.nan, dtype=float, columns=columns, \
                                    index=pd.date_range(start=str_to_date(mintime), end=str_to_date(maxtime), freq='H'))

        for tn in rel_tables:
            cmd = "SELECT {1},time,type,area FROM {0}".format(tn, datacol)
            if yearly_tables:
                yr = tn.split('_')[-1]

                if yr == yr1 and yr == yr2:
                    # use both time limits
                    icnd = conditions + [start_cnd, end_cnd]
                elif yr == yr1:
                    # use lower time limit
                    icnd = conditions + [start_cnd]
                elif yr == yr2:
                    # use upper time limit
                    icnd = conditions + [end_cnd]
                else:
                    icnd = conditions
            else: # only one table -> use time conditions
                icnd = conditions + [start_cnd, end_cnd]

            for idx, cnd in enumerate(icnd):
                if idx == 0:
                    cmd += f" WHERE {cnd}"
                else:
                    cmd += f" AND {cnd}"

            # print(cmd)
            c.execute(cmd)

            for row in c.fetchall():
                d[row[3]].at[str_to_date(row[1]), row[2]] = row[0]

        # correct time index, utc-cet
        if time_format == 'CET':
            for a in d:
                d[a].index = d[a].index + datetime.timedelta(hours=1)

        conn.close()
        return d

    def select_wind_forecast_wrap(self,areas=['SE1', 'SE2', 'SE3', 'SE4', 'DK1', 'DK2', 'FI'],starttime='20180101:00',endtime='20180107:23'):
        """ Get aggregated wind forecast and production for areas """

        categories = ['Wind onsh', 'Wind offsh']

        # # get forecast data
        fdata = self.select_data(table='day_ahead_forecast', areas=areas, categories=categories, starttime=starttime,
                               endtime=endtime, fix_time_range=True)

        # get actual production
        pdata = self.select_data(table='gen_per_type', areas=areas, categories=categories, starttime=starttime,
                               endtime=endtime, fix_time_range=True)

        data = {}
        for a in fdata:
            data[a] = pd.DataFrame(index=fdata[areas[0]].index, columns=['pred', 'actual'])
            data[a]['pred'] = fdata[a].sum(axis=1)
            data[a]['actual'] = pdata[a].sum(axis=1)

        return data

    def add_price_data(self,data_path = 'G:/Master Thesis/Master Thesis/Files/Cap_temp/'):
        import csv

        # make sqlite database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        c.execute('DROP TABLE IF EXISTS spotprice')
        c.execute('CREATE TABLE spotprice (' + \
                  'time TEXT NOT NULL,' + \
                  'area TEXT NOT NULL,' + \
                  'EUR REAL' + \
                  ')')
        #  time stamp in the files is in UTC.
        files = [ \
            'NL_spot_2017.csv', \
            'NL_spot_2018.csv', \
            'PL_spot_2017.csv', \
            'PL_spot_2018.csv'
            ]

        for data_file in files:
            print("Reading {0}".format(data_file))
            area = data_file.split('_')[0]
            with open(data_path + data_file) as f:
                csv_reader = csv.reader(f)
                ridx = 0
                check_each_row = False
                for row in csv_reader:
                    if ridx == 0:
                        # find the currency
                        if 'EUR' in row[1]:
                            currency = 'EUR'
                        elif 'PLN' in row[1]:
                            currency = 'PLN'
                        else:
                            check_each_row = True

                    else:
                        dmyh = row[0].split(' - ')[0]
                        # missing data
                        if 'N/A' in row[1]:
                            price = ''
                        else:
                            price = float(row[1].split(' ')[0])
                            if check_each_row:
                                currency = row[1].split(' ')[1]

                            if currency == 'PLN':
                                price = price*0.23    # exchange rate

                        # since time is in UTC
                        dmyh = datetime.datetime.strptime(dmyh, '%d.%m.%Y %H:%M') + datetime.timedelta(hours=1)
                        dmyh = dmyh.strftime('%Y%m%d:%H')
                        if price != '':
                            cmd = 'INSERT INTO spotprice (time,area,EUR) values("{0}","{1}",{2})'.format(dmyh, area, price)
                            c.execute(cmd)
                    ridx += 1

        conn.commit()
        conn.close()

    def download_flow_data(self,data_type='flow',startyear=2017,endyear=2019,nconnections=None,max_tries=5):
        """ Download flow/exchange data and put into tables
        Also download capacity data
        """

        print(f'--- Downloading {data_type} data ---')
        import time
        # get connections
        cdf = self.get_connections(data_type=data_type)
        if nconnections is not None:
            if type(nconnections) is list:
                cdf = cdf.loc[nconnections,:]
            else:
                cdf = cdf.iloc[range(nconnections),:]

        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        created_tables = []
        nconn = cdf.__len__()
        tstart = time.time()
        for i in cdf.index:
            if i > 1:
                elapsed_time = time.time() - tstart
                avg_time = elapsed_time / (i-1)
                est_time = hour_min_sec((nconn - (i-1)) * avg_time)
                print(f"Estimated time remaining: {est_time[0]} hour {est_time[1]} min")
            cstr = f'{cdf.at[i,"from"]}_{cdf.at[i,"to"]}'
            for y in range(startyear, endyear + 1):
                print(f"Downloading data for {cstr} for {y}")
                for m in range(1,13):
                    # first get data in forward direction
                    for j in range(2): # j = 0: forward, j = 1: backward
                        if j == 0:
                            area1=cdf.at[i,'to']
                            area2=cdf.at[i,'from']
                        else:
                            area1=cdf.at[i,'from']
                            area2=cdf.at[i,'to']

                        d = get_entsoe_exchange_data(area1=area1,area2=area2, data_type=data_type,
                                                     year=y, month=m,max_tries=max_tries)
                        if d is not None:
                            for ts in d:
                                if data_type == 'exchange':
                                    if ts['contract_MarketAgreement.type'] == 'A01':
                                        table_type = 'dayahead'
                                    else:
                                        table_type = 'intraday'
                                else:
                                    table_type = data_type
                                if ts['Period'].index.freq == '15min':
                                    tformat = '%Y%m%d:%H%M'  # quarterly resolution
                                else:
                                    tformat = '%Y%m%d:%H'  # hourly resolution
                                for row in ts['Period'].iteritems():
                                    timestamp = row[0].strftime(tformat)
                                    if data_type == 'capacity':
                                        # capacity data exists in both directions
                                        table_name = f'{table_type}_{cstr}_{j}_{_time2table_(timestamp,data_type)}'
                                    else:
                                        table_name = f'{table_type}_{cstr}_{_time2table_(timestamp,data_type)}'
                                    if table_name not in created_tables:
                                        _create_table_(c,table_name,data_type)
                                        created_tables.append(table_name)

                                    val = row[1]
                                    if j == 0 or data_type == 'capacity': # enter all values
                                        cmd = f"INSERT INTO {table_name} (time,value) values('{timestamp}','{val}')"
                                        _execute_(c,cmd)
                                    elif val > 0: # for backward direction, only enter positive values
                                        # get old value
                                        cmd = f"SELECT value FROM {table_name} WHERE time == '{timestamp}'"
                                        _execute_(c,cmd)
                                        for r in c.fetchall():
                                            old_val = r[0]
                                            break
                                        val = old_val - val
                                        # delete old value
                                        cmd = f"DELETE FROM {table_name} WHERE time == '{timestamp}'"
                                        _execute_(c,cmd)
                                        cmd = f"INSERT INTO {table_name} (time,value) values('{timestamp}','{val}')"
                                        _execute_(c,cmd)
                conn.commit()

        # delete duplicate entries from tables
        for t in created_tables:
            cmd = f"DELETE FROM {t} WHERE ROWID NOT IN (" + \
                  f"SELECT min(ROWID) FROM {t} GROUP BY time)"
            _execute_(c,cmd)
        conn.commit()
        conn.close()


    def select_flow_data(self,connections=['SE1->SE2','SE2->SE3','SE1->SE3'],
                         starttime='20181230:00',endtime='20190101:23',
                         cet_time=False,table='flow',
                         area_sep='->',print_output=True,ctab_name='connections',
                         drop_na_col=False):

        """
        Select exchange data from database. The table types are:
        flow - physical flows
        dayahead - day ahead exchanges
        intraday - intraday exchanges
        capacity - day ahead capacities (always positive in specified direction)
        """

        # connections=['SE1->SE2','SE2->SE1','SE1->FI','SE3->SE4']
        # starttime='20170101:00'
        # endtime='20171231:23'
        # cet_time=False
        # table='capacity'
        # area_sep='->'
        # table: flow, dayahead, intraday, capacity

        #%%
        if table in ['dayahead','intraday']:
            data_type = 'exchange'
        else:
            data_type = table

        conn_list = self.get_connections(data_type,tab_name=ctab_name)
        # map given connection to connection in database
        conn2dbc = {}
        conn_mult = {}
        for conn in connections:
            areas = conn.split(area_sep)
            ind = None
            mult = None
            for ri in conn_list.index:
                if conn_list.at[ri,'from'] == areas[0] and conn_list.at[ri,'to'] == areas[1]:
                    mult = 1
                    ind = ri
                elif conn_list.at[ri,'from'] == areas[1] and conn_list.at[ri,'to'] == areas[0]:
                    mult = -1
                    ind = ri

            if ind is None:
                if print_output:
                    print(f'Connection {conn} does not exist in data')
            else:
                conn2dbc[conn] = ind
                conn_mult[conn] = mult
        # reverse mapping
        dbc2conn = {}
        if data_type == 'capacity':
            dbc_mult = {}
            for dbc in conn_list.index:
                for conn in conn2dbc:
                    if conn2dbc[conn] == dbc:
                        if dbc not in dbc2conn:
                            dbc2conn[dbc] = [conn]
                            dbc_mult[dbc] = [conn_mult[conn]]
                        else:
                            dbc2conn[dbc].append(conn)
                            dbc_mult[dbc].append(conn_mult[conn])
        else:
            for conn in conn2dbc:
                dbc2conn[conn2dbc[conn]] = conn
        #%%
        if cet_time:
            starttime = cet2utc(starttime)
            endtime = cet2utc(endtime)

        time_idx = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq='H')

        # %% find relevant tables
        sqlite_conn = sqlite3.connect(self.db)
        cursor = sqlite_conn.cursor()

        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        cursor.execute(cmd)
        # tables of right type
        rel_tables = [t[0] for t in cursor.fetchall() if f'{table}_' in t[0]]
        rel_years = [f'{yr}' for yr in range(time_idx[0].year,time_idx[-1].year+1)]

        # find all tables to read, and map tables to connections
        get_tables = []
        table2dbc = {}
        table2conn = {}
        for t in rel_tables:
            # all tables for connections in dbc, for given years
            for i in dbc2conn:
                conn_name = f"{conn_list.at[i,'from']}_{conn_list.at[i,'to']}"
                if conn_name in t and t.split('_')[-1] in rel_years:
                    if data_type == 'capacity':
                        # check if both forward and reverse directions have been requested
                        include = True
                        if (1 not in dbc_mult[i] and '_0_' in t) or (-1 not in dbc_mult[i] and '_1_' in t):
                            include = False
                        if include:
                            # find first connection that matches this table
                            get_tables.append(t)
                            for c in conn2dbc:
                                if conn2dbc[c] == i:
                                    if conn_mult[c] == 1 and '_0_' in t:
                                        table2conn[t] = c
                                    elif conn_mult[c] == -1 and '_1_' in t:
                                        table2conn[t] = c
                    else:
                        get_tables.append(t)
                        table2dbc[t] = i

        # %% initialize dataframe
        data = pd.DataFrame(index=time_idx, columns=list(conn2dbc.keys()), dtype=float)

        # %% get data
        # print(get_tables)
        for t in get_tables:
            if data_type == 'capacity':
                conn_name = table2conn[t]
                mult = 1
            else:
                conn = table2dbc[t]
                conn_name = dbc2conn[conn]
                mult = conn_mult[conn_name]

            cmd = f"SELECT time,value FROM {t} WHERE time >= '{starttime}' AND time <= '{endtime}'"
            cursor.execute(cmd)
            for row in cursor.fetchall():
                data.at[str_to_date(row[0]), conn_name] = mult * row[1]

        sqlite_conn.close()
        if cet_time:
            data.index = data.index + datetime.timedelta(hours=1)

        if drop_na_col:
            data.dropna(axis=1,how='all',inplace=True)
        return data

    def get_connections(self,data_type='flow',tab_name='connections'):

        cmd = f"SELECT * FROM {tab_name}_{data_type}"

        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        c.execute(cmd)
        l_conn = []
        for row in c.fetchall():
            l_conn.append(row)

        df_conn = pd.DataFrame(dtype=object,index=[row[0] for row in l_conn],columns=['from','to'])
        df_conn['from'] = [row[1] for row in l_conn]
        df_conn['to'] = [row[2] for row in l_conn]

        conn.close()
        return df_conn

    def define_connections(self,data_type='flow',areas=['SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5',
        'FI','DK1','DK2','EE','LV','LT','RU','PL','DE','NL','GB','FR','BY','BE','CZ','CH','SK','IE','UA'],
                           tab_name='connections'):

        """
        Make table "connections_{type}" with numbered connections that exist in the entso-e transparency data

        :param data_type - entsoe data type, flow/exchange, flow - actual physical flows,
                                                            exchange - day ahead/intraday exchanges

        :return:None
        """
        from time import sleep
        # Find all possible connections

        month = 1
        year = 2020
        nareas = areas.__len__()
        connections = []
        for i in range(nareas):
            for j in range(i+1,nareas):
                connections.append((areas[i],areas[j]))
        # Check which connections has data
        connections_exist = []
        for c in connections:
            df = get_entsoe_exchange_data(area1=c[0],area2=c[1],month=month,year=year,data_type=data_type)
            if df is not None:
                connections_exist.append(c)

        #%% Enter connections into database

        # Create table
        tab_name = f'{tab_name}_{data_type}'

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute(f'DROP TABLE IF EXISTS {tab_name}')
        cmd = f'CREATE TABLE {tab_name} (' + \
              'cindex INTEGER PRIMARY KEY,' + \
              'area1 TEXT NOT NULL,' + \
              'area2 TEXT NOT NULL' + \
              ')'
        c.execute(cmd)

        # Enter data into table
        for idx,(a1,a2) in enumerate(connections_exist):
            cmd = f"INSERT INTO {tab_name} (cindex,area1,area2) values({idx+1},'{a1}','{a2}')"
            c.execute(cmd)

        conn.commit()
        conn.close()

    def update_connections(self,data_type='flow',tab_name='connections'):
        """ Update table with all connections, based on existing tables in database """
        # data_type = 'flow'
        # tab_name = 'connections'
        df_conn = self.get_connections(data_type=data_type,tab_name=tab_name)
        conn_exists = [(df_conn.at[i,'from'],df_conn.at[i,'to']) for i in df_conn.index]

        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        tables = [t for t in get_tables(c) if t.split('_')[0] == data_type]

        tname = f"{tab_name}_{data_type}"
        cmd = f"SELECT max(cindex) from {tname}"
        _execute_(c,cmd)
        cindex = c.fetchone()[0] + 1
        for t in tables:
            a1 = t.split('_')[1]
            a2 = t.split('_')[2]
            if (a1,a2) not in conn_exists:
                if (a2,a1) in conn_exists:
                    print(f'Reverse connection {a2}-{a1} exist in database')
                else:
                    print(f'Adding connection {a1}-{a2} to connection table')
                    conn_exists.append((a1,a2))

                    cmd = f"INSERT INTO {tname} (cindex,area1,area2) values ({cindex},'{a1}','{a2}')"
                    _execute_(c,cmd)
                    cindex += 1
        conn.commit()

    def select_spotprice(self, categories=[], starttime='', endtime=''):
        """ Select time series from sqlite database with entsoe price data. Data
        is returned as a pandas dataframe, and optionally exported to excel file.

        Input:
            areas - list of areas to choose, or in the case of table 'exchange'
                    the list of transfers, by default all categories are selected
            start - string with starting date in format "YYYYMMDD:HH"
            end - string with ending date in format "YYYYMMDD:HH"

        Output:
            pd_data - pandas dataframe with one column for each time series
        """
        if not type(starttime) is str or starttime.__len__() != 11 or starttime[8] != ':':
            print("Error: starttime must be of format 'YYYYMMDD:HH'".format())
            return None
        if not type(endtime) is str or endtime.__len__() != 11 or endtime[8] != ':':
            print("Error: starttime must be of format 'YYYYMMDD:HH'".format())
            return None
        if str_to_date(starttime) > str_to_date(endtime):
            print('Error, start time should be lesser than end time!')
            return None

        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        if categories == []:
            cmd = "SELECT DISTINCT area FROM spotprice"
            c.execute(cmd)
            columns = [val[0] for val in c.fetchall()]
        else:
            columns = categories

        str_categories = '('
        for idx, cat in enumerate(columns):
            if idx > 0:
                str_categories += ",'{0}'".format(cat)
            else:
                str_categories += "'{0}'".format(cat)
        str_categories += ')'

        start_time_cnd = "time >= '{0}'".format(starttime)
        end_time_cnd = "time <= '{0}'".format(endtime)

        cmd = "SELECT * from spotprice where area in" + str_categories + "and " + start_time_cnd + "and " + end_time_cnd
        c.execute(cmd)

        # dataframe for storage
        pd_data = pd.DataFrame(data=np.nan, dtype=float, columns=columns,
                               index=pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq='H'))

        for row in c:
            pd_data.at[str_to_date(row[0]),row[1]] = row[2]

        conn.close()

        return pd_data

    def add_reservoir_data(self,areas=['SE1','SE2'],start_year=2015,end_year=2020):
        """ Add reservoir data """

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        # start_year = 2015
        # end_year = 2020
        tfmt = '%Y%m%d:%H'
        # area='SE1'

        for area in areas:
            print(f'Downloading reservoir data for {area}')
            has_data = False
            table_name = f'reservoir_{area}'

            cmd = f"DROP TABLE IF EXISTS {table_name}"
            _execute_(cursor,cmd)

            cmd = f"CREATE TABLE {table_name} (time TEXT NOT NULL, value REAL NOT NULL)"
            _execute_(cursor,cmd)
            conn.commit()

            for year in range(start_year,end_year+1):
                pass
                res = get_reservoir_data(area,year)
                if res is not None:
                    ts_info,data = res[0],res[1]

                    for time,val in data:
                        # note: YYYYMMDD:22/21 values are due to summer time, replace with :23
                        tstr = time.strftime(tfmt).replace(':22',':23').replace(':21',':23')
                        cmd = f"INSERT INTO {table_name} (time,value) VALUES ('{tstr}',{val/1e3})"
                        _execute_(cursor,cmd)
                        if not has_data:
                            has_data = True
                    conn.commit()

            if not has_data:
                cmd = f"DROP TABLE IF EXISTS {table_name}"
                _execute_(cursor,cmd)

    def select_reservoir_wrap(self,starttime='20180101:00',endtime='20180108:00',areas=['SE1','SE2'],
                              offset=168,cet_time=False,normalize=False,prt=False):
        """
        Select reservoir data, and subtract minimum reservoir level from all values

        rel_values: Return reservoir values normalized with maximum level
        """
        pass

        df = self.select_reservoir_data(starttime,endtime,areas,offset,cet_time)
        # fix certain values manually
        for (a,t,v) in reservoir_fixes:
            dt = datetime.datetime.strptime(t,'%Y%m%d:%H') + datetime.timedelta(hours=offset)
            if cet_time:
                dt += datetime.timedelta(hours=1)
            if a in df.columns and dt in df.index:
                if prt:
                    print(f'Fix reservoir value for {a} for {t}')
                df.at[dt,a] = v


        max_vals = self.select_max(table_type='reservoir',areas=areas)
        if normalize:
            for a in df.columns:
                if a in reservoir_capacity:
                    df[a] = df[a] / reservoir_capacity[a]
                else:
                    df[a] = df[a] / max_vals[a] # don't have capacity use maximum value from database

        return df

    def select_reservoir_data(self,starttime='20180101:00',endtime='20180108:00',areas=['SE1','SE2'],offset=168,cet_time=False):
        """
        Get reservoir values that will contain all hours within time range
        Note: according to ENTSO-E data description, the values are the average filling rates during the week,
        for which the starting time has been entered into the database. Thus the time should be incremented by 84 hours
        to give instantaneous value in the middle of the week. However, when calculating reservoir inflows for SE, a much
        better agreement with data is obtained if the reservoir values are assumed to be the values at the end of the
        given week. Thus, we assume the latter, meaning that the time stamps in the database are incremented by 7 days.

        """
        # starttime = '20180103:00'
        # endtime = '20181229:12'
        # areas = ['SE1','SE2']

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        # find relevant tables
        cmd = "SELECT name FROM sqlite_master WHERE type='table';"
        _execute_(cursor,cmd)

        rel_tables = [r[0] for r in cursor.fetchall() if r[0].split('_')[1] in areas and r[0].split('_')[0] == 'reservoir']

        if starttime.__len__() > 8:
            tfmt = '%Y%m%d:%H'
        else:
            tmft = '%Y%m%d'
        tfmt_db = '%Y%m%d:%H'

        tstart = datetime.datetime.strptime(starttime,tfmt)
        tend = datetime.datetime.strptime(endtime,tfmt)

        # add 8 days at each end of time period for buffer, to make sure returned data covers whole period
        dbstart = (tstart - datetime.timedelta(days=8) - datetime.timedelta(hours=offset)).strftime(tfmt_db)
        dbend = (tend + datetime.timedelta(days=8) - datetime.timedelta(hours=offset)).strftime(tfmt_db)

        # get time values
        table_name = 'reservoir_SE1'

        cmd = f"SELECT DISTINCT time FROM {table_name} WHERE time >= '{dbstart}' AND time <= '{dbend}'"
        _execute_(cursor,cmd)
        str_times = [t[0] for t in cursor.fetchall()]
        str_times.sort()

        index = [datetime.datetime.strptime(t,tfmt_db) for t in str_times]

        df = pd.DataFrame(dtype=float,
                          columns=areas,
                          index=pd.date_range(start=index[0],end=index[-1],freq='7D'))

        for table_name in rel_tables:
            area = table_name.split('_')[1]
            cmd = f"SELECT time,value FROM {table_name} WHERE time >= '{dbstart}' AND time <= '{dbend}'"
            _execute_(cursor,cmd)
            for r in cursor.fetchall():
                time = datetime.datetime.strptime(r[0],tfmt_db)
                df.at[time,area] = r[1]

        # increment index so that values can be interpreted as instantaneous reservoir values
        # df.index += datetime.timedelta(hours=168//2)
        df.index += datetime.timedelta(hours=offset)
        if cet_time:
            df.index += datetime.timedelta(hours=1)

        # drop columns without data
        df.dropna(axis=1,how='all',inplace=True)
        return df

    def select_max(self,table_type='reservoir',areas=('SE1','SE2'),get_min=False):
        """
        Get maximum values for given category, e.g. for using maximum filling values as reservoir capacity
        Works for tables with name of format "{table_type}_{area}"
        """

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        # find relevant tables
        cmd = "SELECT name FROM sqlite_master WHERE type='table';"
        _execute_(cursor,cmd)

        rel_tables = [r[0] for r in cursor.fetchall() if r[0].split('_')[1] in areas and r[0].split('_')[0] == table_type]

        df = pd.Series(index=areas,dtype=float)
        for a in areas:
            rel_a_tables = [t for t in rel_tables if a in t]
            max = 0.0
            for t in rel_a_tables:
                if get_min:
                    cmd = f'SELECT min(value) FROM {t}'
                else:
                    cmd = f'SELECT max(value) FROM {t}'
                _execute_(cursor,cmd)
                for r in cursor.fetchone():
                    val = r
                    if val > max:
                        max = val
            df.at[a] = max
        return df

    def select_inflow_data(self,starttime='2018:01',endtime='2018:01',areas=['SE1','SE2'],table='inflow',wd=weekDef,date_index=False):
        """
        Get inflow data using proper iso week definitions
        """
        if starttime.__len__() > 7:
            w1 = wd.date2week(starttime,str_out=True)
            w2 = wd.date2week(endtime,str_out=True)
        else:
            # time in week format: YYYY:WW
            w1 = starttime
            w2 = endtime

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        # find relevant tables
        cmd = "SELECT name FROM sqlite_master WHERE type='table';"
        _execute_(cursor,cmd)

        rel_tables = [r[0] for r in cursor.fetchall() if r[0].split('_')[1] in areas and r[0].split('_')[0] == table]

        index = wd.range2weeks(w1,w2)

        df = pd.DataFrame(dtype=float,columns=areas,index=index)

        for table_name in rel_tables:
            area = table_name.split('_')[1]
            cmd = f"SELECT time,value FROM {table_name} WHERE time >= '{w1}' AND time <= '{w2}'"
            _execute_(cursor,cmd)
            for r in cursor.fetchall():
                if r[0] in df.index:
                    df.at[r[0],area] = r[1]

        if date_index:
            # set index as date indicating start of week
            idx = pd.date_range(start=wd.week2date(df.index[0]),
                                 end=wd.week2date(df.index[-1]),
                                 freq='7D')
            if idx.__len__() != df.shape[0]:
                raise ValueError('Incorrect length of date index! - indicates bug')
            df.index = idx

        # drop missing columns
        df.dropna(how='all',axis=1,inplace=True)

        return df

    def select_capacity_wrap(self,year=2020,areas=['FI','DK1']):
        """ Select capacity per type, add SE data from other source """

        # def get_entsoe_capacity(year=2020,areas=['FI','DI'],db='D:/Data/entsoe_gen_capacity.db'):
        # db = entsoe_transparency_db.Database(db=db)

        df = self.select_cap_per_type_year(areas,year)
        cap_df = pd.DataFrame(dtype=float,index=areas,columns=list(entsoe_type_map.keys()))
        for a in areas:
            for g in entsoe_type_map:
                types = [gg for gg in entsoe_type_map[g] if gg in df.columns]
                if types:
                    cap_df.at[a,g] = df.loc[a,types].sum()

        se_cap = get_se_capacity()
        se_areas = [a for a in ['SE1','SE2','SE3','SE4'] if a in areas]
        gtypes = [g for g in entsoe_type_map]
        cap_df.loc[se_areas,gtypes] = se_cap.loc[se_areas,gtypes]
        return cap_df

def get_request(query,max_tries=5,timeout=5):
    itry = 0
    while itry < max_tries:
        try:
            r = requests.get(query,timeout=timeout)
            return r
        except requests.exceptions.ConnectionError:
            print(f'Connection error {itry}')
            itry += 1
        except requests.exceptions.Timeout:
            print(f'Timeout {itry}')
            itry += 1
    print(f'Request failed, maximum attempts {max_tries} reached')
    return None

def get_entsoe_gen_data(datatype = 1,area = 'SE1',start='20160101',end='20160101',file=None,max_tries=10):
    """ Get generation data (actual generation or installed capacity) from 
    ENTSO-E transparency database.
    Input:
        datatype - 1: capacity per type
                   2: capacity per unit
                   3: actual generation per type
                   4: actual generation per unit
        area - the price area for which to obtain data
        start - start date
        end - end date
        file - name of xml file to write
    Output:
        data - list containing the returned time series
    Notes: 
    1. For type 4, it is only possible to extract one day of data at a time.
    Only data corresponding to the start date will be returned. For the other 
    types maximum 1 year of data can be obtained at once. 
    2. For type 1 and 2 the data has yearly frequency. For type 3 and 4 the 
    data has hourly frequency.
    """
    
    req_par = {}
    req_url = "https://transparency.entsoe.eu/api?"

    
    if datatype == 1: # Installed capacity per type
        req_par['documentType'] = 'A68'
        req_par['processType'] = 'A33'
    elif datatype == 2: # Installed capacity per unit
         req_par['documentType'] = 'A71'
         req_par['processType'] = 'A33'        
    elif datatype == 3: # Actual generation per type
        req_par['documentType'] = 'A75'
        req_par['processType'] = 'A16'
    elif datatype == 4: # Actual generation per unit
        req_par['documentType'] = 'A73'
        req_par['processType'] = 'A16'
    elif datatype == 5: # Wind generation forecasts
        req_par['documentType'] = 'A69'
        req_par['processType'] = 'A01'
    else:
        print('Wrong data type ''%i'''.format(datatype))
        return None

    req_par['In_Domain'] = tbidz_key[area]
    
    sdate = datetime.datetime(int(start[0:4]),int(start[4:6]),int(start[6:8]))
    edate = datetime.datetime(int(end[0:4]),int(end[4:6]),int(end[6:8]))
    
    
    req_par['periodStart'] = start+'0000'
    if datatype == 4:
        # can only obtain one day of data
        #req_par['periodEnd'] = start+'2300'
        edate = sdate + datetime.timedelta(days=1)
    else:
        #req_par['periodEnd'] = end+'2300'
        edate += datetime.timedelta(days=1)
    req_par['periodEnd'] = edate.strftime("%Y%m%d")+'0000'
        
    #print(edate.strftime("%Y%m%d"))
    send_par = []
    for f in req_par.keys():
        if not req_par[f] == '':
            send_par.append(f)
            
    query = req_url + 'securityToken=' +  req_token + '&'
    for i,f in enumerate(send_par):
        query = query + f + "=" + req_par[f]
        if i < send_par.__len__()-1:
            query = query + '&'

    # max_tries = 5
    itry = 0
    success = False
    while itry < max_tries:
        try:
            r = requests.get(query,timeout=5)
            success = True
            break
        except requests.exceptions.ConnectionError:
            # print(f'Connection error {itry}')
            itry += 1
        except requests.exceptions.Timeout:
            # print(f'Timeout {itry}')
            itry += 1

    if not success and itry == max_tries:
        print('Maximum number of tries reached')
        return None

    if success:
        root = ElementTree.fromstring(r.content)

        # extract prefix
        idx = root.tag.find('}')
        doctype = root.tag[0:idx+1]

        if not file is None:
            tree = ElementTree.ElementTree(root)
            tree.write(file)

        if r.status_code == requests.codes.ok:
            # query was ok

            time_series = root.findall(doctype+'TimeSeries')
            data = []
            for t in time_series:
                ts = {}
                # read fields
                for e in t:

                    field = e.tag[idx+1:]
                    if field == 'Period':
                        # process data
                        start = e.findall(doctype+'timeInterval/'+doctype+'start')[0].text
                        end = e.findall(doctype+'timeInterval/'+doctype+'end')[0].text
                        resolution = e.findall(doctype+'resolution')[0].text
                        resolution_key = {'PT60M':'H','P1Y':'Y','PT15M':'15min','PT30M':'30min'}

                        # create panda time series
                        edate = datetime.datetime(int(end[0:4]),int(end[5:7]),int(end[8:10]))
                        # if yearly resolution, set start date one year before end date
                        if resolution_key[resolution] == 'Y':
                            sdate = edate + datetime.timedelta(days=-365)
                        else:  # hourly/quarterly resolution, take hours into account
                            edate = datetime.datetime(year=int(end[0:4]),month=int(end[5:7]),day=int(end[8:10]),hour=int(end[11:13]),minute=int(end[14:16]))
                            sdate = datetime.datetime(year=int(start[0:4]),month=int(start[5:7]),day=int(start[8:10]),hour=int(start[11:13]),minute=int(start[14:16]))

                        alen = e.__len__() - 2
                        dates = pd.date_range(start=sdate,end=edate,freq=resolution_key[resolution])
                        dates = dates[0:alen] # remove last time, as we only want starting time for each period
                        # print(alen)
                        # print(sdate)
                        # print(edate)
                        # print(start)
                        # print(end)
                        # print(dates)
                        ts['Period'] = pd.Series(np.zeros(alen,dtype=float),index=dates)
                        for (i,point) in enumerate(e[2:]):
                            ts['Period'][i] = float(point[1].text)
                    elif field == 'MktPSRType':
                        # print(e[0].text)
                        ts[field] = e[0].text
                        system_resource = e.findall(doctype+'PowerSystemResources')
                        if not system_resource == []:
                            ts['id'] = system_resource[0][0].text
                            ts['name'] = system_resource[0][1].text

                    else:
                        ts[field] = e.text
                data.append(ts)

                for d in data:
                    if 'MktPSRType' in d:
                        d['production_type'] = tpsr_rabbrv[d['MktPSRType']]
            if data == []:
                print(f'No data for {area} for {start}')
                return None
        else:

            errormsg = root.findall(doctype+'Reason/'+doctype+'text')
            if not errormsg == []:
                print('Invalid query: '+ errormsg[0].text)
            else:
                print('Could not find <Reason> in xml document')
            return None
    else:
        print(f'Request unsuccessful for area {area} and period {start}-{end}')
        return None

    return data


def get_entsoe_exchange_data(area1='SE1',area2='SE2',month=1,year=2018,data_type='exchange',file=None,
                             data_path = 'D:/Data/ENTSO-E/exchanges',max_tries=5):
    """
    document_type:
    flows - A11
    exchanges - A09; returns contract_market types: A01 - day ahead, A05 - intra-day
    capacity - A31;
    """
    # area1 = 'SE1'
    # area2 = 'SE2'
    #
    # month = 1
    # year = 2018
    # file = 'flows'

    if file is not None:
        data_path = Path(data_path)
        data_path.mkdir(exist_ok=True,parents=True)

    if area1 == 'DE':
        if year >= 2019 or (year == 2018 and month >= 10):
            area1 = 'DE_LU'
        else:
            area1 = 'DE_AT_LU'

    if area2 == 'DE':
        if year >= 2019 or (year == 2018 and month >= 10):
            area2 = 'DE_LU'
        else:
            area2 = 'DE_AT_LU'


    startdate = datetime.datetime(year=year, month=month, day=1)
    if month == 12:
        enddate = datetime.datetime(year=year + 1, month=1, day=1)
    else:
        enddate = datetime.datetime(year=year, month=month + 1, day=1)

    # if data_type == 'flow':
    start = startdate.strftime('%Y%m%d%H%M')
    end = enddate.strftime('%Y%m%d%H%M')
    # else:
    #     start = startdate.strftime('%Y%m%d%H%M')
    #     end = (enddate + datetime.timedelta(hours=-24)).strftime('%Y%m%d%H%M')
    # print(start)
    # print(end)
    req_par = {}
    req_url = "https://transparency.entsoe.eu/api?"

    if data_type == 'exchange':
        req_par['documentType'] = 'A09'
    elif data_type == 'flow':
        req_par['documentType'] = 'A11'
    elif data_type == 'capacity':
        req_par['documentType'] = 'A31'
        req_par['Contract_MarketAgreement.Type'] = 'A01'
        req_par['Auction.Type'] = 'A01'
    else:
        print(f'Unknown data type {data_type}')
        return None

    req_par['In_Domain'] = tbidz_key[area1]
    req_par['Out_Domain'] = tbidz_key[area2]

    req_par['periodStart'] = start
    req_par['periodEnd'] = end

    send_par = []
    for f in req_par.keys():
        if not req_par[f] == '':
            send_par.append(f)

    query = req_url + 'securityToken=' + req_token + '&'
    for i, f in enumerate(send_par):
        query = query + f + "=" + req_par[f]
        if i < send_par.__len__() - 1:
            query = query + '&'

    r = get_request(query,max_tries=max_tries)

    if r is not None:
        try:
            root = ElementTree.fromstring(r.content)
            # extract prefix
            idx = root.tag.find('}')
            doctype = root.tag[0:idx + 1]

            if not file is None:
                tree = ElementTree.ElementTree(root)
                tree.write(data_path / f'{file}.xml')

            if r.status_code == requests.codes.ok:
                # query was ok

                time_series = root.findall(doctype + 'TimeSeries')
                data = []
                for t in time_series:
                    ts = {}
                    # read fields
                    for e in t:

                        field = e.tag[idx + 1:]
                        if field == 'Period':
                            # process data
                            start = e.findall(doctype + 'timeInterval/' + doctype + 'start')[0].text
                            end = e.findall(doctype + 'timeInterval/' + doctype + 'end')[0].text
                            resolution = e.findall(doctype + 'resolution')[0].text
                            resolution_key = {'PT60M': 'H', 'P1Y': 'Y', 'PT15M': '15min', 'PT30M':'30min'}

                            edate = datetime.datetime(year=int(end[0:4]), month=int(end[5:7]), day=int(end[8:10]),
                                                      hour=int(end[11:13]), minute=int(end[14:16]))
                            sdate = datetime.datetime(year=int(start[0:4]), month=int(start[5:7]), day=int(start[8:10]),
                                                      hour=int(start[11:13]), minute=int(start[14:16]))
                            alen = e.__len__() - 2

                            dates = pd.date_range(start=sdate, end=edate, freq=resolution_key[resolution])
                            dates = dates[0:alen]  # remove last time, as we only want starting time for each period

                            ts['Period'] = pd.Series(np.zeros(alen, dtype=float), index=dates)
                            for (i, point) in enumerate(e[2:]):
                                ts['Period'][i] = float(point[1].text)
                        else:
                            ts[field] = e.text
                    data.append(ts)
                return data
            else:
                errormsg = root.findall(doctype + 'Reason/' + doctype + 'text')
                if not errormsg == []:
                    print('Invalid query: ' + errormsg[0].text)
                else:
                    print('Could not find <Reason> in xml document')
                return None
        except:
            print(f'Request unsuccessful for {area1}-{area2} for Y{year}M{month}')
            if hasattr(r,'reason'):
                print(f'Reason: {r.reason}')
    else:
        print(f'Request unsuccessful for {area1}-{area2} for Y{year}M{month}')
        return None


def get_entsoe_price_data(area='DE',year=2019,month=12,file=None):
    """
    Get price data for one price area and month

    :param area: price area, note that for germany there have been historically DE_AT_LU and DE_LU
    :param year: year for which to recieve data
    :param month: month for which to recieve data
    :param file: filename, to write recieved xml file
    :return: data, list with one dict for each time series recieved (one time series per day)
    """
    # %% get price data
    # area = 'DE_LU'
    # year = 2019
    # month = 12
    # file = 'prices.xml'
    if area == 'DE':
        if year >= 2019 or (year == 2018 and month >= 10):
            area = 'DE_LU'
        else:
            area = 'DE_AT_LU'


    startdate = datetime.datetime(year=year, month=month, day=1)
    if month == 12:
        enddate = datetime.datetime(year=year + 1, month=1, day=1)
    else:
        enddate = datetime.datetime(year=year, month=month + 1, day=1)
    start = startdate.strftime('%Y%m%d')
    end = (enddate + datetime.timedelta(hours=-24)).strftime('%Y%m%d')

    req_par = {}
    req_url = "https://transparency.entsoe.eu/api?"

    req_par['documentType'] = 'A44'
    req_par['In_Domain'] = tbidz_key[area]
    req_par['Out_Domain'] = tbidz_key[area]

    # sdate = datetime.datetime(int(start[0:4]), int(start[4:6]), int(start[6:8]))
    edate = datetime.datetime(int(end[0:4]), int(end[4:6]), int(end[6:8]))

    req_par['periodStart'] = start + '0000'

    req_par['periodEnd'] = edate.strftime("%Y%m%d") + '0000'

    send_par = []
    for f in req_par.keys():
        if not req_par[f] == '':
            send_par.append(f)

    query = req_url + 'securityToken=' + req_token + '&'
    for i, f in enumerate(send_par):
        query = query + f + "=" + req_par[f]
        if i < send_par.__len__() - 1:
            query = query + '&'


    r = get_request(query)

    if r is None:
        str_month = str(month).zfill(2)
        print(f'Request unsuccessful for {area} for {year}{str_month}')
        return None
    else:
        root = ElementTree.fromstring(r.content)

        # extract prefix
        idx = root.tag.find('}')
        doctype = root.tag[0:idx + 1]

        if not file is None:
            tree = ElementTree.ElementTree(root)
            tree.write(file)

        if r.status_code == requests.codes.ok:
            # query was ok

            time_series = root.findall(doctype + 'TimeSeries')
            data = []
            for t in time_series:
                ts = {}
                # read fields
                for e in t:

                    field = e.tag[idx + 1:]
                    if field == 'Period':
                        # process data
                        start = e.findall(doctype + 'timeInterval/' + doctype + 'start')[0].text
                        end = e.findall(doctype + 'timeInterval/' + doctype + 'end')[0].text
                        resolution = e.findall(doctype + 'resolution')[0].text
                        resolution_key = {'PT60M': 'H', 'P1Y': 'Y', 'PT15M': '15min'}

                        edate = datetime.datetime(year=int(end[0:4]), month=int(end[5:7]), day=int(end[8:10]),
                                                  hour=int(end[11:13]), minute=int(end[14:16]))
                        sdate = datetime.datetime(year=int(start[0:4]), month=int(start[5:7]), day=int(start[8:10]),
                                                  hour=int(start[11:13]), minute=int(start[14:16]))
                        alen = e.__len__() - 2

                        dates = pd.date_range(start=sdate, end=edate, freq=resolution_key[resolution])
                        dates = dates[0:alen]  # remove last time, as we only want starting time for each period

                        ts['Period'] = pd.Series(np.zeros(alen, dtype=float), index=dates)
                        for (i, point) in enumerate(e[2:]):
                            ts['Period'][i] = float(point[1].text)
                    else:
                        ts[field] = e.text
                data.append(ts)
                if data == []:
                    str_month = str(month).zfill(2)
                    print(f'No data for {area} for {year}{str_month}')
                    return None

        else:

            errormsg = root.findall(doctype + 'Reason/' + doctype + 'text')
            if not errormsg == []:
                print('Invalid query: ' + errormsg[0].text)
            else:
                print('Could not find <Reason> in xml document')
            return None

    return data

def get_entsoe_load_data(area='SE1',year=2019,month=1,file=None):

    #%% get load data

    # area = 'SE1'
    # year = 2019
    # month = 1
    # file = 'load.xml'

    startdate = datetime.datetime(year=year, month=month, day=1)
    if month == 12:
        enddate = datetime.datetime(year=year + 1, month=1, day=1)
    else:
        enddate = datetime.datetime(year=year, month=month + 1, day=1)
    start = startdate.strftime('%Y%m%d')
    end = enddate.strftime('%Y%m%d')

    req_par = {}
    req_url = "https://transparency.entsoe.eu/api?"

    req_par['documentType'] = 'A65'
    req_par['processType'] = 'A16'
    req_par['OutBiddingZone_Domain'] = tbidz_key[area]
    req_par['periodStart'] = start + '0000'
    req_par['periodEnd'] = end + '0000'

    send_par = []
    for f in req_par.keys():
        if not req_par[f] == '':
            send_par.append(f)

    query = req_url + 'securityToken=' + req_token + '&'
    for i, f in enumerate(send_par):
        query = query + f + "=" + req_par[f]
        if i < send_par.__len__() - 1:
            query = query + '&'

    r = get_request(query)

    if r is not None:
        root = ElementTree.fromstring(r.content)

        # extract prefix
        idx = root.tag.find('}')
        doctype = root.tag[0:idx + 1]

        if not file is None:
            tree = ElementTree.ElementTree(root)
            tree.write(file)

        if r.status_code == requests.codes.ok:
            # query was ok

            time_series = root.findall(doctype + 'TimeSeries')
            data = []
            for t in time_series:
                ts = {}
                # read fields
                for e in t:

                    field = e.tag[idx + 1:]
                    if field == 'Period':
                        # process data
                        start = e.findall(doctype + 'timeInterval/' + doctype + 'start')[0].text
                        end = e.findall(doctype + 'timeInterval/' + doctype + 'end')[0].text
                        resolution = e.findall(doctype + 'resolution')[0].text
                        resolution_key = {'PT60M': 'H', 'P1Y': 'Y', 'PT15M': '15min', 'PT30M': '30min'}

                        edate = datetime.datetime(year=int(end[0:4]), month=int(end[5:7]), day=int(end[8:10]),
                                                  hour=int(end[11:13]), minute=int(end[14:16]))
                        sdate = datetime.datetime(year=int(start[0:4]), month=int(start[5:7]), day=int(start[8:10]),
                                                  hour=int(start[11:13]), minute=int(start[14:16]))
                        alen = e.__len__() - 2

                        dates = pd.date_range(start=sdate, end=edate, freq=resolution_key[resolution])
                        dates = dates[0:alen]  # remove last time, as we only want starting time for each period

                        ts['Period'] = pd.Series(np.zeros(alen, dtype=float), index=dates)
                        for (i, point) in enumerate(e[2:]):
                            ts['Period'][i] = float(point[1].text)
                    else:
                        ts[field] = e.text
                data.append(ts)

        else:

            errormsg = root.findall(doctype + 'Reason/' + doctype + 'text')
            if not errormsg == []:
                print('Invalid query: ' + errormsg[0].text)
            else:
                print('Could not find <Reason> in xml document')
            return None
    else:
        print(f'Request unsuccessful for area {area} and period {start}-{end}')
        return None

    return data

def get_reservoir_data(area='SE1',year=2015,file=None):
    """
    Get entsoe reservoir data. The data is returned as list with tuples: (start_time,value) where start_time
    is the start of the week for the corresponding data point. The value given is the average reservoir filling
    rate during the week.

    """
    #%% get reservoir data

    # area='SE1'
    # year=2015
    # file='reservoir.xml'

    startdate = datetime.datetime(year,1,1)
    enddate = datetime.datetime(year,12,31)
    dfmt = '%Y%m%d'
    start = startdate.strftime(dfmt)
    end = enddate.strftime(dfmt)

    req_par = {}
    req_url = "https://transparency.entsoe.eu/api?"

    req_par['documentType'] = 'A72'
    req_par['processType'] = 'A16'
    req_par['In_Domain'] = tbidz_key[area]
    req_par['periodStart'] = start + '0000'
    req_par['periodEnd'] = end + '0000'

    send_par = []
    for f in req_par.keys():
        if not req_par[f] == '':
            send_par.append(f)

    query = req_url + 'securityToken=' + req_token + '&'
    for i, f in enumerate(send_par):
        query = query + f + "=" + req_par[f]
        if i < send_par.__len__() - 1:
            query = query + '&'

    r = get_request(query)
    if r is None:
        print(f'Request unsuccessful for {area} for {year}')
        return None
    else:
        root = ElementTree.fromstring(r.content)

        # extract prefix
        idx = root.tag.find('}')
        doctype = root.tag[0:idx + 1]

        if not file is None:
            tree = ElementTree.ElementTree(root)
            tree.write(file)


        if r.status_code == requests.codes.ok:
            # query was ok
            time_series = root.findall(doctype + 'TimeSeries')
            data = []
            get_info = True
            ts_info = {}
            for t in time_series:
                # read fields
                for e in t:

                    field = e.tag[idx + 1:]
                    if field == 'Period':
                        # process data
                        tsstart = e.findall(doctype + 'timeInterval/' + doctype + 'start')[0].text
                        # tsend = e.findall(doctype + 'timeInterval/' + doctype + 'end')[0].text
                        # tsresolution = e.findall(doctype + 'resolution')[0].text
                        timestamp = datetime.datetime.strptime(tsstart, '%Y-%m-%dT%H:%MZ')

                        for (i, point) in enumerate(e[2:]):
                            data.append((timestamp,float(point[1].text)))
                    else:
                        if get_info:
                            if field not in ts_info:
                                ts_info[field] = e.text
                            else:
                                get_info = False
        else:

            errormsg = root.findall(doctype + 'Reason/' + doctype + 'text')
            if not errormsg == []:
                print('Invalid query: ' + errormsg[0].text)
            else:
                print('Could not find <Reason> in xml document')
            return None

        return (ts_info,data)

class DatabaseOutageData():

    def __init__(self, dbase='G:/Master Thesis/Master Thesis/Files/Databases/entsoe_outage.db'):
        self.db = dbase

    def map_ntc2events(self, start_time, end_time, columns=[]):

        event_id_df = pd.DataFrame(data='', dtype=object, columns=columns,
                                   index=pd.date_range(start=str_to_date(start_time), end=str_to_date(end_time),
                                                       freq='H'))

        event_df = self.select_data(start_time=start_time, end_time=end_time, categories=columns)

        event_dict = {}

        for connection in columns:
            for row in event_df[connection]:

                # make the event_dict
                temp = copy.deepcopy(row)
                del temp['event_id']
                event_dict[row['event_id']] = temp

                # add values to the dataframe
                for j in row['periods']:
                    if j['connection'] == connection:
                        start = j['start_time']
                        stop = j['stop_time']
                        start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
                        stop = datetime.datetime.strptime(stop, '%Y-%m-%d %H:%M:%S')
                        # if start time is say 11:15, it will be represented as starting at 11:00
                        if start.minute > 0:
                            start = start.replace(minute=0)
                        else:
                            pass
                        # if end time is say, 11:15, it will be represented as ending at 12:00
                        if stop.minute > 0:
                            stop = stop.replace(minute=0)
                        else:
                            stop = stop - datetime.timedelta(hours=1)
                        start = start.strftime('%Y-%m-%d %H:%M:%S')
                        stop = stop.strftime('%Y-%m-%d %H:%M:%S')

                        event_id_df.loc[start:stop, connection] = event_id_df.loc[start:stop, connection] \
                                                                  + str(row['event_id']) + ';'

        # for connection in columns:
        #     for index,row in event_id_df.iterrows():
        #         temp = row[connection].split(';')
        #         temp_set = set()
        #         temp_set.update([int(temp[x]) for x in range(len(temp)-1)])
        #         row[connection] = list(temp_set)

        return event_id_df, event_dict

    def select_data(self, start_time = '20190101:00', end_time = '20191231:23', categories = []):

        if not type(start_time) is str or start_time.__len__() != 11 or start_time[8] != ':':
            print("Error: starttime must be of format 'YYYYMMDD:HH'".format())
            return None

        if not type(end_time) is str or end_time.__len__() != 11 or end_time[8] != ':':
            print("Error: starttime must be of format 'YYYYMMDD:HH'".format())
            return None

        start_time = str_to_date(start_time)
        end_time = str_to_date(end_time)
        end_time = end_time + datetime.timedelta(hours=1)

        if start_time > end_time:
            print('Error, start time should be lesser than end time!')
            return None

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c2 = conn.cursor()

        # check for empty categories
        if categories == []:
            cmd = "SELECT DISTINCT connection FROM entsoe_outage_data"
            c.execute(cmd)
            categories = [val[0] for val in c.fetchall()]
        else:
            pass

        keys1 = ['event_id','event_start','event_end','asset_id','asset_name','asset_location','asset_type','reason','periods']
        keys2 = ['connection','available_MW','start_time','stop_time']
        data = {}

        for transfer in categories:
            cmd = "SELECT * FROM entsoe_outage_data WHERE event_end >= '{0}' AND connection = '{1}'".format(start_time, transfer)
            c.execute(cmd)
            eventList = []
            previous_event_id = 0

            for count,row in enumerate(c):
                if not count:
                    pass
                else:
                    if previous_event_id == row[0]:
                        continue
                    else:
                        pass
                previous_event_id = row[0]

                periodList = []
                event_id = row[0]
                conn = row[1]
                asset_id = row[2]
                asset_name = row[3]
                asset_location = row[4]
                asset_type = row[5]
                start_date = row[6]
                end_date = row[7]
                available_MW = row[8]
                reason = row[9]
                event_start = row[10]
                event_end = row[11]

                # if event start date > end date, not in range (continue)
                if datetime.datetime.strptime(event_start, '%Y-%m-%d %H:%M:%S') > end_time:
                    continue

                cmd = "SELECT * FROM entsoe_outage_data WHERE event_start == '{0}' AND asset_id = '{1}' AND asset_name = '{2}'" \
                      " AND asset_location = '{3}' AND reason = '{4}' AND event_end = '{5}'".format(event_start,asset_id,asset_name, asset_location,reason,event_end)
                c2.execute(cmd)

                for row2 in c2:
                    periodList.append(dict(zip(keys2, [conn, available_MW, start_date, end_date])))

                eventList.append(dict(zip(keys1, [event_id, event_start, event_end, asset_id, asset_name, asset_location, asset_type, reason, periodList])))

            data[transfer] = eventList

        return data

    def create_database(self,data_path = "F:\hoho"):

        import os

        # connect to database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        # create table
        c.execute('DROP TABLE IF EXISTS entsoe_outage_data')
        cmd = "CREATE TABLE entsoe_outage_data (event_id INTEGER, connection TEXT  NOT NULL, asset_id TEXT NOT NULL,asset_name TEXT NOT NULL, asset_location TEXT NOT NULL," \
              " asset_type TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL,available_MW real,reason TEXT NOT NULL, event_start TEXT NOT NULL, event_end TEXT NOT NULL)"
        c.execute(cmd)

        path_names = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".xml"):
                    path_names = path_names + [os.path.join(root, file)]

        count = 0
        asset_type_dict = {'B21':'AC Link', 'B22': 'DC Link', 'B23': 'Substation','B24':'Transformer'}
        for path in path_names:
            print('Processing {0}'.format(path))
            tree = ElementTree.parse(path)
            root = tree.getroot()

            idx = root.tag.find('}')
            doctype = root.tag[0:idx + 1]

            docStatus = root.findall(doctype + 'docStatus/' + doctype + 'value')

            if docStatus is None:   # if no info available, do nothing
                pass
            else:
                if len(docStatus) == 0:     # if no info available, do nothing
                    pass
                else:
                    if docStatus[0].text != "A05":  # if it is canceled or withdrawn, process next file
                        continue

            time_series = root.findall(doctype+'TimeSeries')
            if time_series is None:     # if time series is of type None, process the next file
                continue
            else:
                for j in time_series:
                    count = count + 1

                    asset_id = ''
                    asset_name = ''
                    asset_location = ''
                    asset_type = ''
                    in_domain = ''
                    out_domain = ''
                    event_start = ''
                    event_end = ''

                    # reason is the last element in the TimeSeries tree
                    reason = j[len(j)-1].findall(doctype + 'text')
                    if (reason is not None) and (len(reason) > 0):
                        reason = reason[0].text
                        reason = str(reason).replace("'","")
                    else:
                        reason = ''

                    for i in j:
                        field = i.tag[idx + 1:]

                        if field == 'in_Domain.mRID':
                            in_domain = i.text

                        elif field == 'out_Domain.mRID':
                            out_domain = i.text

                        elif field == 'start_DateAndOrTime.date':
                            event_start = i.text

                        elif field == 'start_DateAndOrTime.time':
                            event_start = event_start + ' ' + str(i.text)[:-1]
                            event_start = datetime.datetime.strptime(str(event_start),'%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=1)

                        elif field == 'end_DateAndOrTime.date':
                            event_end = i.text

                        elif field == 'end_DateAndOrTime.time':
                            event_end = event_end + ' ' + str(i.text)[:-1]
                            event_end = datetime.datetime.strptime(str(event_end), '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=1)

                        elif field == 'Asset_RegisteredResource':
                            asset_id = asset_id + i.findall(doctype + 'mRID')[0].text + ';'
                            asset_name = asset_name + i.findall(doctype + 'name')[0].text + ';'
                            asset_location = asset_location + i.findall(doctype + 'location.name')[0].text + ';'
                            asset_type = asset_type + asset_type_dict[i.findall(doctype + 'asset_PSRType.psrType')[0].text] + ';'


                        elif field == 'Available_Period':
                            # process data
                            start = i.findall(doctype + 'timeInterval/' + doctype + 'start')[0].text
                            end = i.findall(doctype + 'timeInterval/' + doctype + 'end')[0].text
                            resolution = i.findall(doctype + 'resolution')[0].text
                            position = i.findall(doctype + 'Point/' + doctype + 'position')
                            quantity = i.findall(doctype + 'Point/' + doctype + 'quantity')

                            temp = str(start).split('T')
                            temp_stdate = temp[0].split('-')
                            temp_sttime = temp[1].split('Z')
                            temp_sttime = temp_sttime[0].split(':')

                            temp = str(end).split('T')
                            temp_eddate = temp[0].split('-')
                            temp_edtime = temp[1].split('Z')
                            temp_edtime = temp_edtime[0].split(':')

                            if resolution == 'PT60M':
                                delta = 60
                            elif resolution == 'PT30M':
                                delta = 30
                            elif resolution =='PT15M':
                                delta = 15
                            elif resolution =='PT1M':
                                delta = 1
                            else:
                                print(resolution)   # to get to know and add it to the elif chain
                                exit(-1)

                            for k in range(len(position)):
                                minute_delta_start = delta*(int(position[k].text)-1)
                                start_date = datetime.datetime(int(temp_stdate[0]),int(temp_stdate[1]),int(temp_stdate[2]),int(temp_sttime[0]),int(temp_sttime[1])) + datetime.timedelta(minutes=minute_delta_start)

                                if k != (len(position)-1):
                                    minute_delta_end = delta * (int(position[k+1].text) - 1)
                                    end_date = datetime.datetime(int(temp_stdate[0]),int(temp_stdate[1]),int(temp_stdate[2]),int(temp_sttime[0]),int(temp_sttime[1])) + datetime.timedelta(minutes=minute_delta_end)
                                else:
                                    end_date = datetime.datetime(int(temp_eddate[0]),int(temp_eddate[1]),int(temp_eddate[2]),int(temp_edtime[0]),int(temp_edtime[1]))

                                # dates are in UTC, convert to CET (UTC + 1)
                                start_date = start_date + datetime.timedelta(hours=1)
                                end_date = end_date + datetime.timedelta(hours=1)

                                cmd = "INSERT INTO entsoe_outage_data(event_id,connection, asset_id, asset_name, asset_location," \
                                      " asset_type, start_date, end_date, available_MW, reason, event_start, event_end) values ({0},'{1}','{2}','{3}'" \
                                      ",'{4}','{5}','{6}','{7}',{8},'{9}','{10}','{11}')".format(count,tbidz_rkey[out_domain] + ' - ' + tbidz_rkey[in_domain], asset_id[:-1], asset_name[:-1],
                                                                               asset_location[:-1], asset_type[:-1], start_date, end_date, float(quantity[k].text), reason, event_start, event_end)
                                c.execute(cmd)
        # delete identical data
        cmd = "DELETE FROM entsoe_outage_data WHERE rowid NOT IN (SELECT min(rowid) FROM entsoe_outage_data GROUP BY " \
              "connection, asset_id, asset_name, asset_location, start_date, end_date, available_MW, reason, event_start, event_end)"
        c.execute(cmd)

        conn.commit()
        conn.close()

def get_entsoe_outage_data(data_path='F:\hoho',start_year=2016,end_year=2019,areas = ['SE','DK','NO','SE1','SE2','SE3','SE4','DK1','DK2','NO1','NO2','NO3','NO4','NO5','FI','EE','LT','LV','DE_CA','NL','RU','PL',],overwrite=False):
    """
    Download ENTSO-E transparency data: A78 Unavailability of transmission infrastructure
    DocumentType: A78
    Article: 10.1.a and 10.1.b

    NOTE:
    Data returned is in the form of zip files containing xml documents. Function searches for data
    for connections connecting any two areas. For each connection A1-A2 a new folder will be made, containing
    all xml files with outage data for that connection. The files are named by year and month,
    e.g. 2019M10_001-001-[...].xml, where the first part of the name is added to distinguish files extracted
    from different zip files. Since the maximum number of xml files that can be downloaded in a single zip
    file is 200, and this number is sometimes exceeded if data is downloaded for a whole year, the function
    downloads separate zip files for each month in a year, extracts them and renames them according to this
    format.

    In the get request, the In_Domain and Out_Domain should both be codes to different EIC Control areas.
    Note that EIC codes for control areas (CA) and Market Bidding Areas (MBA) and countries may be different.
    Eic codes can be found at:
    https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_areas
    The codes are stored in tbidz_key, and hence the areas should use the keys defined in this dict.

    :param:data_path - folder where xml files are stored, a new folder will be made for each connection
    :param:start_year - year to start searching for outage data
    :param:end_year - last year to search for data
    :param:areas - list with control areas (CTA), function will search for data connecting any two areas
    :param:overwrite - True/False, overwrite old xml files

    :return:None
    """

    import zipfile, io, os
    import calendar
    # data_path = 'D:/Data/ENTSO-E/Transmission unavailability'
    Path(data_path).mkdir(exist_ok=True,parents=True)

    # overwrite = False
    file = 'entsoe.xml'

    # areas = ['SE','DK','NO','FI','EE','LT','LV','DE','PL',]

    req_par = {}
    req_url = "https://transparency.entsoe.eu/api?"

    req_par['documentType'] = 'A78'
    req_par['periodStart'] = ''
    req_par['periodEnd'] = ''
    req_par['In_Domain'] = ''
    req_par['Out_Domain'] = ''

    send_par = [s for s in req_par]

    # years = range(2016,2020)
    years = range(start_year,end_year+1)

    for a1 in areas:
        for a2 in areas:
            if a1 != a2:
                for year in years:
                    print('----------------------------------------')
                    print(f"DOWNLOADING DATA FOR {a2}-{a1} for {year}")
                    print('----------------------------------------')
                    for month in range(1,13): # download one month at a time

                        ndays = calendar.monthrange(year, month)[1]
                        startdate = datetime.datetime(year, month, 1, 0)
                        enddate = datetime.datetime(year, month, ndays, 23)
                        start = startdate.strftime('%Y%m%d%H00')
                        end = enddate.strftime('%Y%m%d%H00')

                        req_par['In_Domain'] = tbidz_key[a1]
                        req_par['Out_Domain'] = tbidz_key[a2]
                        req_par['periodStart'] = start
                        req_par['periodEnd'] = end

                        query = req_url + 'securityToken=' + req_token + '&'
                        for i, f in enumerate(send_par):
                            query = query + f + "=" + req_par[f]
                            if i < send_par.__len__() - 1:
                                query = query + '&'

            #%%
                        r = requests.get(query)

                        if r.status_code == 200:
                            path = Path(data_path)/f"{a2}-{a1}"
                            path.mkdir(exist_ok=True)
                            z = zipfile.ZipFile(io.BytesIO(r.content))
                            z.extractall(path=path)
                            # rename files as "M1_[XXX].xml"
                            for file in z.filelist:
                                newfile = path/ f"{year}M{month}_{file.filename}"
                                if os.path.isfile(newfile):
                                    if overwrite:
                                        os.remove(newfile)
                                        os.rename(path/file.filename,newfile)
                                else:
                                    os.rename(path / file.filename, newfile)

                        else:
                            root = ElementTree.fromstring(r.text)
                            rootname = root.tag.split('{')[1].split('}')[0]
                            ns = {'ns':rootname}
                            reason = root.find('ns:Reason',ns).find('ns:text',ns).text
                            print(f"Found no data for {a2}-{a1}, Month {month}:\n{reason}")



def aggregate_gen_per_type_data(pd_data):
    """ Aggregate production data according to more broader categories given
    in 'aggr_types'. 
    Input:
        pd_data - original data frame with entsoe transparency production types
    Output:
        pd_aggrdata - new data frame with production aggregated by categories
        given in 'aggr_types'
    """

    pd_aggrdata = pd.DataFrame(columns=aggr_types,index=pd_data.index,dtype=float)
    for gentype in aggr_types:
        cols =  list( intersection( list(pd_data.columns), \
                   [tpsr_rabbrv[f] for f in aggr_types[gentype]] ))
        if cols != []:
            pd_aggrdata[gentype] = pd_data[cols].sum(axis=1)
        else:
            pd_aggrdata[gentype] = np.zeros(pd_aggrdata.__len__(),dtype=float)
    
    return pd_aggrdata

def plot_generation(data,area = '',savefigs=False):
    """ Plot the generation in data. Data may be dictionary with one dataframe
        per area, or it may be a single dataframe, in which case the name
        of the area must be specified
    Input:
        data - dictionary with one DataFrame for each price area
    """
    
    import seaborn
    
    if type(data) == dict:
        #figidx = 1
        for area in data.keys():
            
            number_of_plots = data[area].columns.__len__()
        
            colors = seaborn.color_palette("hls",number_of_plots) # Set2, hls
            #colors = seaborn.husl_palette(number_of_plots)
            #colors = seaborn.mpl_palette("Set2",number_of_plots)
            
            plt.figure()
            
            lines = plt.plot(data[area])
            for i,l in enumerate(lines):
                l.set_color(colors[i])
            
            plt.legend(data[area].columns)
            plt.title(area)
            plt.ylabel('MW')
            
            if savefigs:
                plt.savefig("Figures/{0}.png".format(area))
    
            #figidx += 1
    else:

            number_of_plots = data.columns.__len__()
        
            colors = seaborn.color_palette("hls",number_of_plots) # Set2, hls
            #colors = seaborn.husl_palette(number_of_plots)
            #colors = seaborn.mpl_palette("Set2",number_of_plots)
            
            plt.figure()
            
            lines = plt.plot(data)
            for i,l in enumerate(lines):
                l.set_color(colors[i])
            
            plt.legend(data.columns)
            plt.title(area)
            plt.ylabel('MW')
            
            if savefigs:
                plt.savefig("Figures/{0}.png".format(area))


def print_installed_capacity(data,areas=[],file='Data/capacities.txt'):
    """ Print pretty tables with capacity per area and type
    Input: 
        data - dictionary with DataFrame for each area
    """
    if file is not None:
        f = open(file,'w')
        
    if areas == []:
        areas = area_codes
        
    for a in areas:
                
        # find nan and zero columns
        nnan = data[a].isnull().sum()
        nzero = (data[a] == 0).sum()
        nrows = data[a].__len__()
        cols = []
        for c in data[a].columns:
            if nnan[c] + nzero[c] < nrows:
                cols.append(c)
                
        # short column names
        #scols = [tpsr_rabbrv[tpsr_abbrv[c]] for c in cols]
    
        t = PrettyTable()
        t.field_names = ['year'] + list(cols)
        for row in data[a].iterrows():
            t.add_row([row[0]]+[row[1][f] for f in cols])

        print('Area: {0}'.format(a))
        print(t)
        if not file is None:
            f.write('Area: {0}\n'.format(a))
            f.write(t.get_string())
            f.write('\n')
    if not file is None:
        f.close()

def fillna(pdframe):
    """ Replace NaN values with zero for those years in which there is at least
    one column which has non-missing data
    """
    for row in pdframe.iterrows():
        if row[1].isnull().sum() < row[1].__len__():
            # impute zero values  
            pdframe.loc[row[0]] = pdframe.loc[row[0]].fillna(0)
            
def compare_nuclear_generation():
    """ Compare nuclear generation of individual units with aggregate values """
       
    db = DatabaseGenUnit()
    
    starttime = '20180101:00'
    endtime = '20181231:00'
    
    data,plants = db.select_data(start=starttime,end=endtime,countries=['SE','FI'])
    
    pnames = ['Ringhals block 1 G11','Ringhals block 1 G12','Ringhals block 2 G21','Ringhals block 2 G22']
    
    
    # select nuclear
    se_nuclear = [idx for idx in plants.index if plants.at[idx,'type'] == 'Nuclear' and plants.at[idx,'country'] == 'SE']
    fi_nuclear = [idx for idx in plants.index if plants.at[idx,'type'] == 'Nuclear' and plants.at[idx,'country'] == 'FI']
    
    nuclear_list = [idx for idx in plants.index if plants.at[idx,'type'] == 'Nuclear']

    data_nuclear = data.loc[:,nuclear_list].copy(deep=True)
    data_nuclear.columns = [plants.at[code,'name'] for code in data_nuclear.columns]


    data_decom = data_nuclear.loc[:,pnames].copy(deep=True)
    data_decom['tot'] = data_decom.sum(axis=1)
    data_decom.plot()
    
    # get aggregate nuclear production
    db2 = Database()
    data2 = db2.select_gen_per_type_wrap(starttime=starttime,endtime=endtime,areas=['SE3','FI'])
    
    # compare aggregate and individual nuclear production
    plt.figure()
    ax = data2['SE3'].loc[:,'Nuclear'].plot()
    data.loc[:,se_nuclear].sum(axis=1).plot(ax=ax)
    plt.show()
    plt.close()
    
    plt.figure()
    ax = data2['FI'].loc[:,'Nuclear'].plot()
    data.loc[:,fi_nuclear].sum(axis=1).plot(ax=ax)
    plt.show()
    plt.close()
    
    
def normalize_forecast_data(data,cap_margin = 0.95):
    res_data = {}
    #%%
    nstd = 3

    fig_path = 'D:/NordicModel/Results/wind_forecasts/fig'

    Path(fig_path).mkdir(exist_ok=True,parents=True)

    for a in data:
        ts = data[a]['actual']

        # find peaks
        peakvalues,peakidxs = find_peaks(ts,mindist=200)

        # select envelope of increasing peaks
        prev_peak = peakvalues[0]
        idxs = []
        for idx,p in enumerate(peakvalues):
            if p > prev_peak:
                idxs.append(peakidxs[idx])
                prev_peak = p

        # find points corresponding to convex hull

        #%%
        x = np.array(pd.to_timedelta(pd.Series(ts.index[idxs])).dt.total_seconds())
        y = np.array(ts.iloc[idxs])

        hull_idxs = [idxs[i] for i in find_convex_hull(x,y)]

        #%%

        ts_norm = pd.Series(data=0,dtype=float,index=ts.index)

        norm_idx = 2
        norm = ts.iat[idxs[norm_idx]]
        for i in range(ts_norm.__len__()):
            if norm_idx < idxs.__len__()-1 and i >= idxs[norm_idx]:
                norm_idx += 1
                norm = ts.iat[idxs[norm_idx]]
            ts_norm.iat[i] = norm


        ts.plot()
        ts.iloc[peakidxs].plot(style=':',marker='*')
        ts.iloc[idxs].plot(style=':',marker='s')
        ts.iloc[hull_idxs].plot(style=':',marker='o')
        ts_norm.plot()
        plt.legend(['actual production','peaks','increasing peaks','convex hull','normalization'])
        plt.title(f'Normalization for {a}')
        plt.savefig(Path(fig_path) / f'normalization_{a}.png')

        plt.clf()
        (ts / ts_norm * cap_margin).plot()
        plt.title(f'Normalized production {a}')
        plt.savefig(Path(fig_path) / f'normalized_prod_{a}.png')
        plt.clf()

        res_data[a] = ts_norm / cap_margin

    plt.close()
    return res_data


def fit_forecast_model(name='windFit1',areas=['DK1','DK2','FI'],starttime = '20150201:00',endtime = '20181231:23'):

    db = Database()
    import pickle

    # options for data selection and normalization

    fig_path = 'D:/NordicModel/Results/wind_forecasts/fig/'

    (Path(fig_path) / 'bin_plots').mkdir(exist_ok=True,parents=True)
    (Path(fig_path) / 'forecast_error_plots').mkdir(exist_ok=True,parents=True)

    hperday = 24

    nBins = 5
    nPlotBins = 20
    nstd = 4  # removing outliers more than nstd standard deviations from mean

    data_path = 'D:/NordicModel/Results/wind_forecasts/'
    nPeriods = 24  # periods in each scenario

    data = db.select_wind_forecast_wrap(starttime=starttime, endtime=endtime, areas=areas)

    # remove outliers and replace with interpolated values

    for a in data:
        data[a][(data[a] - data[a].mean()).abs() > nstd * data[a].std()] = np.nan
        data[a] = data[a].interpolate()

    # %%
    ndata = normalize_forecast_data(data, cap_margin=0.95)
    for a in data:
        for c in data[a].columns:
            data[a][c] = data[a][c] / ndata[a]

    # %% plot error distribution for different hours

    Path(fig_path).mkdir(exist_ok=True, parents=True)

    estats = {}
    for a in data:
        data[a]['err'] = data[a]['pred'] - data[a]['actual']
        estats[a] = pd.DataFrame(index=range(hperday), columns=['err_abs', 'err_std'])

    for hour in range(hperday):

        idxs = data[areas[0]].index.hour == hour

        for a in data:
            estats[a].at[hour, 'err_abs'] = data[a].loc[idxs, 'err'].abs().mean()
            estats[a].at[hour, 'err_std'] = data[a].loc[idxs, 'err'].std()

            # plot histogram with errors

            data[a].loc[idxs, 'err'].hist()
            plt.title(f'Forecast error for {a} for hour {hour}')
            plt.xlabel('Prediction error (MWh)')
            plt.savefig(Path(fig_path) / Path('forecast_error_plots') / f"hist_forecast_error_{a}_h{hour}.png")
            plt.clf()

    for a in estats:
        estats[a].plot()
        plt.grid()
        plt.title(f'Forecast error for {a}')
        plt.xlabel('Hour')
        plt.savefig(Path(fig_path) / f"forecast_error_{a}")
        plt.clf()

    plt.close('all')

    # %% plot error distribution for different bins

    for area in data:

        c, bins = pd.cut(data[area]['actual'], bins=nBins, retbins=True, labels=range(nBins))

        for bin in range(nBins):
            data[area].loc[c == bin, 'err'].hist(bins=nPlotBins)
            plt.title(f'Error distribution for area {area} and bin {bin}')
            plt.savefig(Path(fig_path) / Path('bin_plots') / f'bin_distr_{area}_b{bin}.png')
            plt.clf()

    # %% fit multivariate normal distribution

    times = [i for i in range(nPeriods)]
    vidx = {}
    for aidx, area in enumerate(areas):
        d = {}
        for t in times:
            d[t] = aidx * times.__len__() + t
        vidx[area] = d
    # reverse mapping: varidx -> area, time
    idx2at = {}
    for a in vidx:
        for t in vidx[a]:
            idx2at[vidx[a][t]] = {'area': a, 'time': t}

    nVars = areas.__len__() * times.__len__()
    nObs = (data[areas[0]].index.hour == 0).sum()

    fit_data = pd.DataFrame(dtype=float, index=range(nObs), columns=range(nVars))
    for area in areas:
        for tidx in times:
            # get all data
            tmpdata = data[area]['err'].loc[data[area]['err'].index.hour == tidx]
            ridx = 0
            for r in tmpdata.iteritems():
                fit_data.at[ridx, vidx[area][tidx]] = r[1]
                ridx += 1

    # %%
    # remove mean from data
    fit_data = fit_data - fit_data.mean()

    cov = np.cov(fit_data, rowvar=0)

    # save parameters of fit
    wind_model = {
        'name': name,
        'areas': areas,
        'nPeriods': nPeriods,
        'startime': starttime,
        'endtime': endtime,
        'cov': cov,
        'cap': ndata,
    }

    with open(Path(data_path) / f'{name}.pkl', 'wb') as f:
        pickle.dump(wind_model, f)


    # plot visualization of covariance matrix
    stds = [cov[i][i] for i in range(cov.__len__())]

    plt.plot(stds)

    plot_cov_idx = []
    for a in areas:
        for lag in range(2):
            idx = vidx[a][lag]
            plot_cov_idx.append(idx)
            plt.plot(cov[idx])


    plt.grid()
    plt.legend(['std']+[f"cov[{n}]" for n in plot_cov_idx])
    plt.xlabel('variable')
    plt.title('Covariance matrix')

    plt.savefig(Path(fig_path)/f"cov_model_{name}.png")

    plt.close()

def generate_wind_scenarios(name='windFit1',nSamp = 20,starttime='20180101:00'):


    db = Database(db='D:/Data/entsoe_transparency.db')
    import pickle

    # options for scenario realization
    # name = 'windFit1'
    data_path = 'D:/NordicModel/Results/wind_forecasts/'
    # nSamp = 20  # number of scenario realizations
    # starttime = '20180101:00'



    with open(Path(data_path)/f"{name}.pkl",'rb') as f:
        wind_model = pickle.load(f)

    cov = wind_model['cov']
    areas = wind_model['areas']
    nPeriods = wind_model['nPeriods']
    timerange = pd.date_range(start=str_to_date(starttime),end=(str_to_date(starttime)+datetime.timedelta(hours=nPeriods-1)),freq='H')
    endtime = timerange[-1].strftime('%Y%m%d:%H')


    # setup index dicts
    times = [i for i in range(nPeriods)]
    vidx = {}
    for aidx,area in enumerate(areas):
        d = {}
        for t in times:
            d[t] = aidx*times.__len__() + t
        vidx[area] = d
    # reverse mapping: varidx -> area, time
    idx2at = {}
    for a in vidx:
        for t in vidx[a]:
            idx2at[vidx[a][t]] = {'area':a,'time':t}

    nVars = areas.__len__() * times.__len__()


    #%% draw samples

    # initialize data and enter true outcome

    # get actual production
    categories = ['Wind offsh','Wind onsh']
    wind_data = db.select_data(table='gen_per_type', areas=areas, categories=categories, starttime=starttime,
                             endtime=endtime, fix_time_range=True)


    rls = {}
    capacity = {}
    for a in wind_data:
        rls[a] = pd.DataFrame(columns=range(nSamp+1),index=timerange)
        capacity[a] = wind_model['cap'][a].at[str_to_date(starttime)]
        rls[a][0] = wind_data[a].sum(axis=1) / capacity[a]


    #%% generate scenarios

    rands = np.random.multivariate_normal(mean=np.zeros(nVars),cov=cov,size=nSamp)

    for sidx in range(nSamp):
        for vidx,err in enumerate(rands[sidx]):
            area = idx2at[vidx]['area']
            # realization = predicted*(1+err)
            realz = np.min([1,np.max([0,rls[area].iat[idx2at[vidx]['time'], 0]+err])])
            rls[area].iat[idx2at[vidx]['time'],sidx+1] = realz

    return rls,capacity,wind_model


def elis_test_function(data_path='F:/'):
    """
    Code for testing databases
    """
    import os
    data_path = Path(data_path)
    data_path.mkdir(exist_ok=True, parents=True)

    filename = 'errors.txt'

    if os.path.exists(data_path / filename):
        os.remove(data_path / filename)

    db = DatabaseOutageData()
    # get_entsoe_outage_data(start_year=2013, end_year=2018)
    # db.create_database()

    starttime = '20130101:00'
    endtime = '20181231:23'

    categories = ['SE1 -  SE2']

    import nordpool_db
    nordpool_db = nordpool_db.Database()
    cap_df, code_df = nordpool_db.select_exchange_capacities(starttime, endtime, categories, reduc_code_flag=True)

    map_ntc, event_list = db.map_ntc2events(cap_df)

    columns = list(map_ntc)
    print('Total connections found')
    print(len(columns))
    file1 = open(data_path / filename,"a")
    for connection in columns:
        for index, row in map_ntc.iterrows():
            capacity = str(cap_df.at[index, connection])

            if len(row[connection]):
                min_capacity = 999999
                for j in row[connection]:
                    s = event_list.get(j)
                    for i in s['periods']:
                        if i['connection'] == connection:
                            min_capacity = min(min_capacity, i['available_MW'])
                min_capacity = str(min_capacity)
            else:
                min_capacity = 'nan'

            # if (code != 1010) and (min_capacity == 'nan'):
            #     print('{0} {1} {2} {3}'.format(index, connection, capacity, min_capacity))

            if (capacity != min_capacity) and (min_capacity != 'nan') and (capacity != 'nan'):
                file1.write('{0} {1} {2} {3}\n'.format(index, connection, capacity, min_capacity))

    file1.close()

def fix_zero_perios(dic,ncols_drop=4,print_output=True):
    """
    For GB (maybe other countries) there are periods with zero values for several categories, i.e. missing data
    This function replaces these zero values with missing values, so that the values can be interpolated. This avoids
    getting erronous zero values when e.g., computing the minimum production values over an extended period of time

    :param dic:
    :return:
    """
    # ncols_drop = 4 # if at least 4 categories drop to zero, mark as possible missing data
    # area = 'GB'
    # navg = 3
    timefmt = "%Y%m%d:%H"
    #%% identify periods with zero values for all categories (except for wind)
    for area,df_raw in dic.items():

        # exclude some categories which often go to zero
        df = df_raw.loc[:,[c for c in df_raw.columns if c not in ['Solar','Hydro pump']]]

        # find drops to zero
        idxs_drop = (df==0) & (df.shift() > 0)
        idxs_incr = (df.shift() == 0) & (df > 0)

        # get indices where drops occur for several categories at once
        idxs_drop_num = np.flatnonzero(idxs_drop.sum(axis=1) >= ncols_drop)
        idxs_incr_num = np.flatnonzero(idxs_incr.sum(axis=1) >= ncols_drop)

        #%%
        fix_period_count = 0
        fix_vals_count = 0
        for i,idx_drop in enumerate(idxs_drop_num):
            # find index where data increase from zero
            idx_incr = idx_drop + 1
            while idx_incr < df.__len__() and idx_incr not in idxs_incr_num:
                idx_incr += 1
            # check that idx_incr is less than next idx_drop
            if (i+1 == idxs_drop_num.__len__() or idx_incr < idxs_drop_num[i+1]) and idx_incr < df.__len__() and (idx_incr > idx_drop + 1):
                drop_cols = df.columns[idxs_drop.iloc[idx_drop,:]]
                incr_cols = df.columns[idxs_incr.iloc[idx_incr,:]]
                # categories must be the same
                if set(drop_cols) == set(incr_cols):
                    # print(f'Period {i}: {df.index[idx_drop].strftime(timefmt)}-{df.index[idx_incr].strftime(timefmt)}')
                    fix_period_count += 1
                    fix_vals_count += idx_incr - idx_drop
                    df_raw.loc[df.index[idx_drop:idx_incr],drop_cols] = np.nan
        if fix_period_count:
            if print_output:
                print(f'{area}: replaced {fix_vals_count} hourly zero values with nan for {fix_period_count} periods')

def fix_gen_data_manual(data_raw,cet_time=False):
    """ Some manual fixes to the entsoe generation data, i.e. remove outliers, replace days with missing data """

    tfmt = '%Y%m%d:%H'

    # drop nuclear for SE4
    if 'SE4' in data_raw:
        data_raw['SE4'].drop(columns=['Nuclear'],inplace=True)
    # fix individual outliers
    # fix: (area,time,type,value)

    for area,tstr,gtype,val in gen_per_type_fixes:
        if area in data_raw:
            t = datetime.datetime.strptime(tstr,tfmt)
            if cet_time:
                t += datetime.timedelta(hours=1)
            if t in data_raw[area].index and gtype in data_raw[area].columns:
                data_raw[area].at[t,gtype] = val

    # fill missing days with data
    for a,gap in miss_gen_data_periods:
        if a in data_raw:
            fill_daily_gap(data_raw[a],gap)

    return data_raw

def fix_time_shift(dic,cet_time=False,print_output=True):
    """ Handle DST shift """
    if cet_time:
        shift_hour = 2
    else:
        shift_hour = 1

    time = dic[list(dic)[0]].index  # assuming same time for all areas
    month, day, wd, hour = time.month, time.day, time.weekday, time.hour
    shift = np.flatnonzero((month == 10) & (wd == 6) & (hour == shift_hour) & (day > 24))  # DST shift hours in October
    for i, df in dic.items():
        # DST shift in October
        for s in shift:
            if sum(df.iloc[s - 1, :]) / sum(df.iloc[s - 2, :]) > 1.5:
                df.iloc[s - 1, :] /= 2
                df.iloc[s, :] = df.iloc[s - 1, :]
                if print_output:
                    print(f'Fixing values for DST shift for {i}')

    return dic

def fix_entsoe(dic, limit=20,cet_time=False,print_output=True,fill_lim=2):
    """ Handle DST shift and interpolate remaining nans.
    Error if time series starts with last sunday in october at 02:00...
    Note: works for hourly data
    """
    if cet_time:
        shift_hour = 2
    else:
        shift_hour = 1

    time = dic[list(dic)[0]].index  # assuming same time for all areas
    month, day, wd, hour = time.month, time.day, time.weekday, time.hour
    shift = np.flatnonzero((month == 10) & (wd == 6) & (hour == shift_hour) & (day > 24))  # DST shift hours in October
    for i, df in dic.items():
        print(i)
        print(df.head())
        # DST shift in October
        for s in shift:

            print(s)
            if sum(np.isnan(df.iloc[s, :])) > 0 and sum(df.iloc[s - 1, :]) / sum(df.iloc[s - 2, :]) > 1.5:
                df.iloc[s - 1, :] /= 2
                df.iloc[s, :] = df.iloc[s - 1, :]

        # Remaining nans
        df.interpolate(limit=limit, inplace=True)  # interpolate up to limit samples
        # fill edges of data
        df.fillna(method='ffill',inplace=True,limit=fill_lim)
        df.fillna(method='bfill',inplace=True,limit=fill_lim)
        nnan = np.sum(np.array(np.isnan(df)))
        if nnan > 0:
            if print_output:
                print(f'Too many ({nnan}) nans for {i} in Entso-e data for %s, interpolate might not work properly.' % i)
            for gtype in df.columns:
                nnan = df[gtype].isna().sum()
                if nnan > 0:
                    pass
                    # print(f'{gtype}: {nnan}')
    return dic

def fill_daily_gap(df,gap=('20160203','20160209'),col=None):
    """ If any of the days in the range given by 'gap' exist in the dataframe, data from the previous or following
    whole day outside the gap will be copied to all of the days in the gap. This is used to fix periods of missing
    data.
    """
    #%% fix gap in LT data for Solar and Wind onsh, by repeating data from previous day
    # area = 'LT'
    # df = dic[area]
    # gap = ('20160203','20160209') # first and last date for which there is a gap in the data
    fill_vals = True
    gap_dt = (str_to_date(gap[0]),str_to_date(gap[1])+datetime.timedelta(hours=23))
    # check if timerange overlaps gap
    if (df.index[0] > gap_dt[1]) or (df.index[-1] < gap_dt[0]): # no overlap
        return None
        # fill_vals = False
    else:
        # check if there is complete 24 hour period before first gap day
        if (gap_dt[0] - df.index[0]).days >= 1:
            # print('Use previous day')
            fill_idxs = pd.date_range(start=gap_dt[0]+datetime.timedelta(hours=-24),
                                      end=gap_dt[0]+datetime.timedelta(hours=-1),
                                      freq=df.index.freq)
        elif (df.index[-1] - gap_dt[1]).days >= 1:
            # print('Use following day')
            fill_idxs = pd.date_range(start=gap_dt[1]+datetime.timedelta(hours=1),
                                      end=gap_dt[1]+datetime.timedelta(hours=24),
                                      freq=df.index.freq)
        else:
            # print(f'Cannot find complete day to fill gap for data')
            # fill_vals = False
            return None
    if fill_vals:
        day_range = pd.date_range(start=str_to_date(gap[0]),end=str_to_date(gap[1]),freq='D')
        for d in day_range:
            if d in df.index and (d+datetime.timedelta(hours=23)) in df.index: # this day exists in df
                miss_idxs = pd.date_range(start=d,end=d+datetime.timedelta(hours=23),freq='H')
                if col is None: # repeat all columns
                    df.loc[miss_idxs,:] = np.array(df.loc[fill_idxs,:])
                else:
                    df.loc[miss_idxs,col] = np.array(df.loc[fill_idxs,col])



def fix_pl_prices(df):
    # fix for PL price, publised in PLN from 2 March 2017 and 19 November 2019
    if 'PL' in df.columns:
        sfmt = '%Y%m%d:%H'
        t1 = datetime.datetime.strptime('20170301:23',sfmt)
        t2 = datetime.datetime.strptime('20191119:22',sfmt)
        tidxs = [t for t in df.index if t >= t1 and t <= t2]
        df.loc[tidxs,'PL'] = df.loc[tidxs,'PL'] * 0.23
        
def plot_graphs_entsoe_thesis(entsoe_db_path='G:/Master Thesis/Master Thesis/Files/Databases/entsoe_outage.db',
                              nordpool_db_path='G:/Master Thesis/Master Thesis/Files/Databases/nordpool.db',
                              fig_path='F:/Graphs/ENTSOE', starttime='20180101:00', endtime='20181231:23',categories=[]):
    """
    :param
        entsoe_db_path : path of entsoe_outage.db
        nordpool_db_path : path of nordpool.db
        fig_path : path of the folder where the figures will be stored
        starttime : in the format 'yyyymmdd:hh'
        endtime : in the format 'yyyymmdd:hh'
    :return
        None
    """
    import nordpool_db

    if not categories:
        # get the list of connections from the nordpool database
        df_cap = nordpool_db.Database(db=nordpool_db_path).select_exchange_capacities(starttime=starttime,endtime=endtime)
        # create a list of connections
        columns = list(df_cap)
    else:
        columns = categories

    # images are saved as png
    # path where figures are stored
    (Path(fig_path)).mkdir(exist_ok=True, parents=True)

    # turn off plotting
    plt.ioff()

    # get the events from the entsoe_outage database
    db = DatabaseOutageData(dbase=entsoe_db_path)
    df_event_id, event_dict = db.map_ntc2events(starttime, endtime, columns=columns)

    list_asset_id = [value['asset_id'] for key, value in event_dict.items()]

    # remove the semicolons
    list_asset_id = list(set([a for x in list_asset_id for a in x.split(';')]))
    while '' in list_asset_id:
        list_asset_id.remove('')

    # create a dictionary with asset_ids and corresponding event_id
    dict_asset_to_event_id = {}

    for asset in list_asset_id:
        for key, value in event_dict.items():
            if asset in value['asset_id']:
                if asset not in dict_asset_to_event_id.keys():
                    dict_asset_to_event_id[asset] = [key]
                else:
                    dict_asset_to_event_id[asset] = dict_asset_to_event_id[asset] + [key]

    # create a dictionary with asset_ids and corresponding asset_names
    dict_asset_to_names = {}

    for asset in list_asset_id:
        for key, value in event_dict.items():
            if asset in value['asset_id']:
                pos = value['asset_id'].split(';').index(asset)
                dict_asset_to_names[asset] = value['asset_name'].split(';')[pos]

    # create a dictionary with asset_ids and corresponding asset_types
    dict_asset_to_types = {}

    for asset in list_asset_id:
        for key, value in event_dict.items():
            if asset in value['asset_id']:
                pos = value['asset_id'].split(';').index(asset)
                dict_asset_to_types[asset] = value['asset_type'].split(';')[pos]

    # for histogram
    asset_type = ['AC Link','DC Link','Substation','Transformer']
    for count,ty in enumerate(asset_type):
        asset_hist = np.array([])
        for key, value in dict_asset_to_event_id.items():
            if dict_asset_to_types[key] != ty:
                continue
            indices = set()
            for conn in columns:
                for x in value:
                    indices.update(df_event_id.index[df_event_id[conn].str.contains(str(x))])
            asset_hist = np.append(asset_hist,[len(indices)])

        fig = plt.figure(count+1)
        plt.hist(asset_hist,bins=np.arange(min(asset_hist), max(asset_hist) + 5, 5))
        plt.grid(True, axis='y', zorder=0)
        plt.title('Histogram of hours of unavailability due to assets of type \'{0}\''.format(ty))
        plt.xlabel('Hours of Unavailability')
        plt.ylabel('Number of Assets')
        plt.savefig(Path(fig_path)  / f'{ty}_histogram_unavailability_hours.png', bbox_inches="tight")
    plt.close('all')

    for conn in columns:

        # create a folder for each connection
        path = Path(fig_path) / f'{conn}'
        path.mkdir(exist_ok=True, parents=True)

        output_hours = {}
        output_hours_by_asset_type = {'AC Link':set(), 'DC Link':set(), 'Substation':set() ,'Transformer':set()}
        # create four dictionaries (list) to hold asset unavailabiliity for each asset type
        output_hours_assets_per_asset_type = {'AC Link':{}, 'DC Link':{}, 'Substation':{} ,'Transformer':{}}

        for key, value in dict_asset_to_event_id.items():
            indices = set()
            for x in value:
                indices.update(df_event_id.index[df_event_id[conn].str.contains(str(x))])

            if not indices:
                continue

            output_hours[dict_asset_to_names[key]] = len(indices)
            output_hours_by_asset_type[dict_asset_to_types[key]].update(indices)
            output_hours_assets_per_asset_type[dict_asset_to_types[key]][dict_asset_to_names[key]] = len(indices)

        for k,v in output_hours_by_asset_type.items():
            output_hours_by_asset_type[k] = len(v)

        od = {k: v for k, v in sorted(output_hours.items(), key=lambda item: item[1])}
        fig = plt.figure(1)
        plt.barh(range(len(od)), od.values(), align='center', zorder=3)
        plt.yticks(range(len(od)), od.keys())
        plt.grid(True, axis='x', zorder=0)
        plt.title('Hours of unavailability on {0} due to each asset'.format(conn))
        plt.xlabel('Hours')
        plt.ylabel('Asset Names')
        plt.savefig(path / f'{conn}_unavailability_hours_per_asset.png', bbox_inches="tight")

        od = {k: v for k, v in sorted(output_hours_by_asset_type.items(), key=lambda item: item[1])}
        fig = plt.figure(2)
        plt.barh(range(len(od)), od.values(), align='center', zorder=3)
        plt.yticks(range(len(od)), od.keys())
        plt.grid(True, axis='x', zorder=0)
        plt.title('Hours of unavailability on {0} due to each asset type'.format(conn))
        plt.xlabel('Hours')
        plt.ylabel('Asset Types')
        plt.savefig(path / f'{conn}_unavailability_hours_per_asset_type.png', bbox_inches="tight")

        fig_no = 2
        for k,v in output_hours_assets_per_asset_type.items():
            fig_no += 1
            od = {m: n for m, n in sorted(v.items(), key=lambda item: item[1])}
            fig = plt.figure(fig_no)
            plt.barh(range(len(od)), od.values(), align='center', zorder=3)
            plt.yticks(range(len(od)), od.keys())
            plt.grid(True, axis='x', zorder=0)
            plt.title('Hours of unavailability on {0} due to assets of type \'{1}\''.format(conn,k))
            plt.xlabel('Hours')
            plt.ylabel('Asset Names')
            plt.savefig(path / f'{conn}_{k}_unavailability_hours.png', bbox_inches="tight")

        plt.close('all')


def germany_negative_prices():

    db = Database(db='D:/Data/entsoe_prices.db')

    data = db.select_price_data(areas=['DE'],starttime='20180101:00',endtime='20191231:23',cet_time=True)

    #%% analyze incidents of negative prices
    neg = data['DE'] < 0
    a = 'DE'
    incidents = [] # list with number of hours with negative prices for each incident (=continuous period with negative prices)
    negative_flag = 0
    hours = 0
    for t in data.index:
        if data.at[t,a] < 0:
            if not negative_flag:
                # start of period with negative prices
                negative_flag = True
                hours = 1
            else:
                # continuation of negative price period
                hours += 1
        else:
            if negative_flag:
                # end of period with negative prices
                incidents.append(hours)
                negative_flag = False
                hours = 0
    # add last incident, if prices are negative at end of time period
    if hours > 0:
        incidents.append(hours)


    #%% calculate share of all negative hours that belong to period of certain length
    ntot = sum(incidents)
    nshare = pd.Series(0.0,index=range(1,max(incidents)+1))
    for len in incidents:
        nshare.at[len] += 1e2*len/ntot

    six_share = 100 - nshare.loc[range(1,6)].sum()
    # share of hours with duration at least 6 hours
    print(f'Share of hours with duration >= 6: {six_share:0.2f}')

    # make histogram
    # plt.hist(incidents)
    # plt.show()
    fig_path = Path('C:/Users/elisn/Box Sync/Python/TNO-Curtailment/Figures')
    fh = 7.5
    fw = 12
    cm_per_inch = 2.5

    f = plt.figure()
    f.set_size_inches(w=fw/cm_per_inch,h=fh/cm_per_inch)
    nshare.plot.bar()

    plt.xlabel('Duration of period [h]')
    plt.ylabel('Share of hours [%]')
    plt.tight_layout()
    # plt.grid()

    plt.savefig(fig_path / f'negative_price_duration.png')
    plt.savefig(fig_path / f'negative_price_duration.eps')

def print_capacity_excel_file():
    """
    Create excel file with capacity per type for EU countries
    :return:
    """
    #%% build database from scratch
    db = Database('D:/Data/entsoe_capacity.db')
    cap_areas = [
        'SE','NO','FI','DK1','DK2','DK','EE','LT','LV','PL','NL','FR','BE',
        'ES','PT','IE','GB','IT','CH','AT','CZ','CH','SK','HU','SI','CR','BL','BH','MK','SR','GR','RO','MT','AL'
    ]
    start_year = 2016
    end_year = 2020

    db.download_cap_per_type_data(start_year=start_year,end_year=end_year,areas=cap_areas)
    # data = get_entsoe_gen_data(datatype=1,area='FR',start='20180101',end='20180101')

    #%%
    df = db.select_cap_per_type_data(areas=None)

    # rename countries
    df = df.rename(columns={a:tbidz_name[a] for a in cap_areas})
    # remove nan values
    df1 = df.dropna(axis=1,how='all')

    #%%
    # print data for only one year
    cols = []
    for col in df1.columns:
        if col[0] not in cols:
            cols.append(col[0])

    writer = pd.ExcelWriter('capacity.xlsx')
    df1.to_excel(excel_writer=writer,sheet_name='all years')

    for year in range(start_year,end_year+1):
        df2 = pd.DataFrame(dtype=float,index=[t for t in tpsr_key],columns=cols)
        for col in df1.columns:
            df2.at[col[1],col[0]] = df1.at[year,col]
        df2 = df2.dropna(axis=0,how='all')
        df2.to_excel(excel_writer=writer,sheet_name=f'{year}')

    writer.close()



def download_svk_data(start_year=2015,end_year=2016):

    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select
    import time
    import os
    from selenium.webdriver.common.keys import Keys


    dl_path = Path('C:/Users/elisn/Downloads/')
    svk_url = 'https://mimer.svk.se/ProductionConsumption/ProductionIndex'

    driver = webdriver.Chrome()

    driver.get(svk_url)
    time.sleep(3)

    for year in range(start_year,end_year+1):
        el_to = driver.find_element_by_id('periodTo')
        el_from = driver.find_element_by_id('periodFrom')

        #%% set FROM to first day of year
        el_from.click()
        tab = driver.find_element_by_class_name('datepicker')
        el_prev = tab.find_element_by_class_name('prev')
        el_next = tab.find_element_by_class_name('next')
        el_month = tab.find_element_by_class_name('datepicker-switch')
        # find current month and decide direction
        if int(el_month.text.split(' ')[1]) < year:
            el_but = el_next
        else:
            el_but = el_prev
        while el_month.text != f'Januari {year}':
            time.sleep(0.01)
            el_but.click()
        for day in tab.find_elements_by_class_name('day'):
            if day.text == '1':
                break
        day.click()
        el_from.click()
        #%% set to to last day of year
        el_to.click()
        tab = driver.find_element_by_class_name('datepicker')
        el_next = tab.find_element_by_class_name('next')
        el_prev = tab.find_element_by_class_name('prev')
        el_month = tab.find_element_by_class_name('datepicker-switch')
        # find current month and decide direction
        if int(el_month.text.split(' ')[1]) > year:
            el_but = el_prev
        else:
            el_but = el_next
        while el_month.text != f'December {year}':
            time.sleep(0.01)
            el_but.click()
        for day in tab.find_elements_by_class_name('day'):
            if day.text == '31':
                break
        day.click()

        #%% download all production types for all areas
        prod_types = {
            'Hydro':'Vattenkraftproduktion',
            'Nuclear':'Kärnkraftproduktion',
            'Thermal':'Övrig värmekraftproduktion',
            'Wind':'Vindkraftproduktion',
            'Solar':'Solenergiproduktion',
            'Other':'Uppmätt ospecificerad produktion',
        }

        areas = {
            'SE1':'Elområde 1',
            'SE2':'Elområde 2',
            'SE3':'Elområde 3',
            'SE4':'Elområde 4',
        }

        for a in areas:
            el_area = driver.find_element_by_id('ConstraintAreaId')
            area_select = Select(el_area)
            area_select.select_by_visible_text(areas[a])

            for t in prod_types:
                el_type = driver.find_element_by_id('ProductionSortId')
                type_select = Select(el_type)
                type_select.select_by_visible_text(prod_types[t])

                time.sleep(0.1)

                el_send = [e for e in driver.find_elements_by_tag_name('input') if e.get_attribute('title')=='Skicka'][0]
                el_send.click()
                time.sleep(0.1)

                # Note: if no data exists no save button exists
                try:
                    el_save = [e for e in driver.find_elements_by_tag_name('a') if e.text == 'Spara som CSV'][0]
                    dl_url = el_save.get_attribute('href')

                    # download file
                    el_save.click()
                    time.sleep(2)
                    #%%
                    # rename file
                    newfile = dl_path/f'SVK_{a}_{t}_{year}.csv'
                    try:
                        # remove old file if it exsits
                        if os.path.exists(newfile):
                            os.remove(newfile)
                        os.rename(dl_path/f'ProduktionsStatistik.csv',newfile)
                    except FileNotFoundError as e:
                        print(f'File {dl_path/"ProduktionsStatistik.csv"} not found')
                except IndexError:
                    print(f'No data exists for {a}:{t}')

def calculate_inflow_data(startyear=2018,endyear=2019,areas=['SE1','SE2'],table='inflow3',offset=0,
                          db_name='inflow.db',data_path='D:/Data',
                          res_db_path='D:/Data/entsoe_reservoir.db',
                          gen_db_path='D:/Data/entsoe_gen.db'):
    """
    Calculate inflow data and put into reservoir database
    Requires entsoe generation database

    :param startyear:
    :param endyear:
    :param areas:
    :param res_db_path:
    :param gen_db_path:
    :return:
    """
    #%% calculate weekly inflows

    from help_functions import interp_time
    # startyear = 2018
    # endyear = 2019
    # areas = ['SE1','SE2']


    res_db = Database(db=res_db_path)
    gen_db = Database(db=gen_db_path)

    inflow = {}

    for year in range(startyear,endyear+1):
        starttime = f'{year}0101:00'
        endtime = f'{year+1}0101:00'

        #%% select reservoir data

        res_df = res_db.select_reservoir_data(starttime=starttime,endtime=endtime,areas=areas,offset=offset)

        #%% select hydro generation data
        gen_data = gen_db.select_gen_per_type_wrap_v2(areas=areas,
                                                      starttime=starttime,
                                                      endtime=endtime,
                                                      dstfix=True)


        inflow_idx = pd.date_range(start=str_to_date(starttime),end=str_to_date(endtime),freq='7D')[:52]
        inflow_df = pd.DataFrame(index=inflow_idx,columns=areas,dtype=float)

        # interpolate instantaneous reservoir values
        res_idx = [i for i in inflow_idx] + [str_to_date(endtime)]
        res_inst_df = interp_time(res_idx,res_df)

        for area in areas:
            print(f'Calculating inflow data for {area} {year}')
            # remove last hour since it belongs to next year
            gen_df = gen_data[area].iloc[:gen_data[area].__len__()-1,:]

            # sum inflows for weeks in each year
            # for each year, the first week is the first 168 hours of the year, and week 52 has all remaining hours
            for i,t in enumerate(inflow_idx):

                if i == inflow_idx.__len__() - 1:
                    tidxs = gen_df.index >= t
                else:
                    tidxs = (gen_df.index >= t) & (gen_df.index < inflow_idx[i+1])
                # calculate inflow:
                # Inflow = res[t+1] - res[t] + sum(weekly hydro)
                inflow_df.at[t,area] = res_inst_df.at[res_idx[i+1],area] - res_inst_df.at[res_idx[i],area] \
                                       + gen_df.loc[tidxs,'Hydro'].sum() / 1e3

        nneg = (inflow_df < 0).sum().sum()
        if nneg > 0:
            max_neg = inflow_df.min().min()
            print(f'{nneg} negative values, min value: {max_neg:0.2f}')
            # inflow_df[inflow_df<0.0] = 0.0

        inflow[year] = inflow_df

    #%% put data into database
    conn = sqlite3.connect(Path(data_path) / db_name)
    cursor = conn.cursor()

    made_tables = []

    for year in inflow:

        for area in inflow[year].columns:

            table_name = f'{table}_{area}'
            for i,val in enumerate(inflow[year][area]):
                if not np.isnan(val):
                    tstr = f'{year}:' + f'{i+1}'.zfill(2)

                    if table_name not in made_tables:
                        cmd = f"DROP TABLE IF EXISTS {table_name}"
                        _execute_(cursor,cmd)
                        cmd = f"CREATE TABLE {table_name} (time TEXT NOT NULL,value REAL NOT NULL)"
                        _execute_(cursor,cmd)
                        conn.commit()
                        made_tables.append(table_name)

                    cmd = f"INSERT INTO {table_name} (time,value) VALUES ('{tstr}',{val})"
                    _execute_(cursor,cmd)
        conn.commit()

def calculate_inflow_data_iso(starttime='20141225:00',endtime='20210105:00',areas=['SE1'],
                              offset=0,table='inflow1',db_name='inflow.db',db_path='D:/Data/',wd=weekDef):
    """
    Calculate inflow data considering ISO week definitions
    NOTE: The problem with this method is that it does not account for spillage. E.g, for NO4 during dec 24-25 2018,
    there was large amounts of spillage, which will give negative calculated inflows...

    :param starttime:
    :param endtime:
    :param areas:
    :return:
    """
    # starttime = '20171225:00'
    # endtime = '20200105:00'
    # areas=['SE1']
    res_db_path='D:/Data/entsoe_reservoir.db'
    gen_db_path='D:/Data/entsoe_gen.db'

    from help_functions import interp_time

    res_db = Database(db=res_db_path)
    gen_db = Database(db=gen_db_path)
    inflow_db = Database(db=Path(db_path) / db_name )

    #%% select reservoir data

    res_df = res_db.select_reservoir_wrap(starttime=starttime,endtime=endtime,areas=areas,offset=offset)

    #%% select hydro generation data
    gen_data = gen_db.select_gen_per_type_wrap_v2(areas=areas,
                                                  starttime=starttime,
                                                  endtime=endtime,
                                                  dstfix=True,
                                                  limit=30)

    # # find weeks in range
    # wrange = (date2week_iso(starttime),date2week_iso(endtime))
    # # windex contains starting hour of each week
    # windex = pd.date_range(start=week2date_iso(wrange[0])[0],end=week2date_iso(wrange[1])[0],freq='7D')

    # wrange = wd.range2weeks(starttime,endtime)
    # # windex contains starting hour of each week
    # wstart = wd.week2date(week=wrange[0])
    # wend = wd.week2date(week=wrange[1])
    # windex = pd.date_range(start=wstart,end=wend,freq='7D')
    # if windex[-1] != wend:
    #     raise ValueError('Wrong index, should end at beginning of new week')
    # list of weeks
    windex = wd.range2weeks(starttime,endtime,sout=True)
    # start of weeks
    tindex = wd.range2weeks(starttime,endtime,sout=False)

    inflow_df = pd.DataFrame(index=windex,columns=areas,dtype=float)

    # interpolate instantaneous reservoir values
    res_inst_df = interp_time(tindex,res_df)

    for area in areas:
        print(f'Calculating inflow data for {area}')
        gen_df = gen_data[area]

        # compute inflows for each week
        for i,w in enumerate(windex):
            # all hours in this week (in hydro data)
            t1 = tindex[i] # start of week
            t2 = tindex[i+1] # start of next week
            tidxs = (gen_df.index >= t1) & (gen_df.index < t2)
            inflow_df.at[w,area] = res_inst_df.at[t2,area] - res_inst_df.at[t1,area] \
                                   + gen_df.loc[tidxs,'Hydro'].sum() / 1e3

    nneg = (inflow_df < 0).sum().sum()
    if nneg > 0:
        max_neg = inflow_df.min().min()
        print(f'{nneg} negative values, min value: {max_neg:0.2f}')
        # inflow_df[inflow_df < 0] = 0.0

    #%% put data into database
    conn = sqlite3.connect(inflow_db.db)
    cursor = conn.cursor()

    made_tables = []
    for area in inflow_df.columns:
        table_name = f'{table}_{area}'
        for w in inflow_df.index:
            val = inflow_df.at[w,area]
            if not np.isnan(val):
                # tstr = date2week_v2(t,ws=ws,pw=pw,sout=True)
                if table_name not in made_tables:
                    cmd = f"DROP TABLE IF EXISTS {table_name}"
                    _execute_(cursor,cmd)
                    cmd = f"CREATE TABLE {table_name} (time TEXT NOT NULL,value REAL NOT NULL)"
                    _execute_(cursor,cmd)
                    conn.commit()
                    made_tables.append(table_name)

                cmd = f"INSERT INTO {table_name} (time,value) VALUES ('{w}',{val})"
                _execute_(cursor,cmd)
        conn.commit()

    return inflow_df

def validate_calculated_inflow_data(inflow_table='inflow1',db='inflow.db',fig_tag='c1',db_path='D:/Data',wd=weekDef):

    #%% select reservoir inflow data
    fig_path = 'D:/NordicModel/Figures'
    db = Database(db=Path(db_path) / db)

    starttime='2015:01'
    endtime='2018:52'
    # starttime = '20171225:00'
    # endtime = '20181231:23'

    areas=['SE1','SE2','SE3','SE4']

    inflow_entsoe = db.select_inflow_data_v2(starttime=starttime,endtime=endtime,areas=areas,table=inflow_table,wd=wd,date_index=True)
    inflow_entsoe['SE'] = inflow_entsoe.sum(axis=1)

    from statistic_calculations import inflow_energiforetagen
    inflow_en = inflow_energiforetagen()
    # inflow_en.index = wd.range2weeks(inflow_en.index[0],inflow_en.index[-1],sout=False)[:-1]
    inflow_en.index = [wd.week2date(t) for t in inflow_en.index]
    # inflow_entsoe.index = [week2date_iso(i)[0] for i in inflow_entsoe.index]
    # inflow_entsoe.index += datetime.timedelta(days=0)
    # inflow_en.index = [week_to_date(i) for i in inflow_en.index]

    # inflow_entsoe.index += datetime.timedelta(days=3)

    tidx = [i for i in inflow_en.index if i in inflow_entsoe.index]

    df1 = inflow_entsoe.loc[tidx,'SE'] / 1e3
    df2 = inflow_en.loc[tidx,'SE'] / 1e3

    rmse = np.mean(np.abs(df1-df2))/np.mean(np.abs(df2))
    rmse2 = np.mean(np.abs(df1.loc[df1.index.year >= 2017]-df2.loc[df2.index.year >= 2017]))/np.mean(np.abs(df2.loc[df2.index.year>=2017]))

    #%%
    f,ax = plt.subplots()
    f.set_size_inches(6,3)

    df1.plot(ax=ax,label='Calculated')
    df2.plot(ax=ax,label='Data energiföretagen')

    ax.grid()
    ax.legend(title=f'Norm. MAE: {rmse:0.4f}')
    # ax.legend(title=f'Norm. MAE: {rmse:0.4f}\nNorm. MAE 2017-2018: {rmse2:0.4f}')

    ax.set_ylabel('TWh/week')
    # plt.title(f'{inflow_table}')

    plt.savefig(Path(fig_path)/ f'inflow_{fig_tag}.png')
    plt.savefig(Path(fig_path)/ f'inflow_{fig_tag}.eps')

def validate_inflow_data(tag='t1',fig_path = 'D:/NordicModel/Figures'):
    """
    Compare inflow data for all sources:
    Calculated (from hydro production and reservoir levels)
    MAF (from ENTSO-E MAF 2020 database)
    Data Energiföretagen (Data from Energiföretagen)

    :return:
    """
    wd = WeekDef(week_start=4,proper_week=True)
    #%% select reservoir inflow data
    # fig_path = 'D:/NordicModel/Figures'
    res_db = Database(db='D:/Data/reservoir.db')
    inflow_db = Database(db='D:/Data/inflow.db')
    inflow_table = 'inflow'

    starttime='2015:02'
    endtime='2018:52'
    # starttime = '20171225:00'
    # endtime = '20181231:23'

    areas=['SE1','SE2','SE3','SE4']

    inflow_entsoe = inflow_db.select_inflow_data(starttime=starttime,endtime=endtime,areas=areas,table=inflow_table)
    inflow_entsoe['SE'] = inflow_entsoe.sum(axis=1)

    #%% get MAF inflow data
    from maf_hydro_data import Database as MafDb
    maf_db = MafDb(db='D:/Data/maf_hydro.db')

    maf_area_map = {
        'SE':['SE01','SE02','SE03','SE04'],
        'NO':['NOS0','NOM1','NON1'],
        'FI':['FI00'],
    }

    maf_areas = ['SE01','SE02','SE03','SE04']
    starttime = '20150101'
    endtime = '20161231'

    res = maf_db.select_data(areas=maf_areas,starttime=starttime,endtime=endtime,htype='RES')

    # aggregate inflow data by countries, for comparison
    inflow_maf = pd.DataFrame(dtype=float,index=res.index,columns=[c for c in maf_area_map])

    for c in ['SE']:
        inflow_maf[c] = res.loc[:,maf_area_map[c]].sum(axis=1)


    #%%
    from statistic_calculations import inflow_energiforetagen
    inflow_en = inflow_energiforetagen()

    # inflow_entsoe.index = [week_to_date(i) for i in inflow_entsoe.index]
    # inflow_en.index = [week_to_date(i) for i in inflow_en.index]
    # inflow_maf.index = [week_to_date(i) for i in inflow_maf.index]

    tidx = [i for i in inflow_en.index if i in inflow_entsoe.index and i in inflow_maf.index]

    df1 = inflow_entsoe.loc[tidx,'SE'] / 1e3
    df2 = inflow_en.loc[tidx,'SE'] / 1e3
    df3 = inflow_maf.loc[tidx,'SE'] / 1e3

    rmse1 = np.mean(np.abs(df1-df2))/np.mean(np.abs(df2))
    rmse2 = np.mean(np.abs(df3-df2))/np.mean(np.abs(df2))

    #%%
    f,ax = plt.subplots()
    f.set_size_inches(6,4)

    df1.plot(ax=ax,label='Calc.',linestyle='--',linewidth=1.5)
    df3.plot(ax=ax,label='MAF',linestyle='-.',linewidth=1.5)
    df2.plot(ax=ax,label='Data',linestyle='-',linewidth=0.5,color='k')

    ax.grid()
    ax.legend(title=f'NMAE (calc): {rmse1:0.2f}\nNMAE (MAF): {rmse2:0.2f}')
    ax.set_ylabel('GWh/week')

    plt.savefig(Path(fig_path)/ f'inflow_validate_{tag}.png')
    plt.savefig(Path(fig_path)/ f'inflow_validate_{tag}.eps')

    return inflow_maf,inflow_entsoe,inflow_en
def compare_entsoe_nordpool_total_production():
    db = Database(db='D:/Data/entsoe_gen.db')
    import nordpool_db
    db_np = nordpool_db.Database(db='D:/Data/nordpool.db')

    areas = ['SE1','SE2','SE3','SE4','FI','DK1','DK2','NO1','NO2','NO3','NO4','NO5','LV','LT','EE']
    # areas
    starttime = '20180101:00'
    endtime = '20181231:23'
    dic = db.select_gen_per_type_wrap_v2(starttime=starttime,endtime=endtime,drop_data=False,areas=areas,drop_pc=95,print_output=True,cet_time=True)
    df_np = db_np.select_data(table='production',starttime=starttime,endtime=endtime,categories=areas)

    #%%
    # db.add_reservoir_data(areas=['LT','LV'],start_year=2015,end_year=2020)
    prod = pd.DataFrame(dtype=float,index=df_np.index,
                        columns=pd.MultiIndex.from_product([areas,['entsoe','nordpool','diff']],names=['area','data']))
    for a in areas:
        prod[(a,'entsoe')] = dic[a]['Tot']
        prod[(a,'nordpool')] = df_np[a]
        prod[(a,'diff')] = (prod[(a,'entsoe')] - prod[(a,'nordpool')]).abs()

    #%%
    # for a in areas:
    #     # plt.figure()
    #     prod.loc[:,(a,slice(None))].plot()

    #%%
    mae = prod.loc[:,(slice(None),'diff')].abs().mean().squeeze()
    print(mae)
    return prod

def drop_non_se_data():
    """ Only keep Swedish production data from SvK """
    db = Database(db='D:/Data/gen.db')

    conn = sqlite3.connect(db.db)
    c = conn.cursor()

    cmd = "SELECT name FROM sqlite_master WHERE type='table'"
    c.execute(cmd)
    # tables of right type
    rel_tables = [t[0] for t in c.fetchall() if 'SE' not in t[0] or t[0].split('_')[-2] == 'SE']
    for t in rel_tables:
        cmd = f'DROP TABLE {t}'
        c.execute(cmd)
    conn.commit()
    conn.close()

def get_se_capacity():
    """ Add SE capacities, not in ENTSO-E
    From Energiföretagen, data for januari 1, 2021
    """
    df = pd.DataFrame(index=['SE1','SE2','SE3','SE4'],columns=['Thermal','Hydro','Wind','Nuclear'],
                      data=[[273,5320,1652,0],[681,8076,3876,0],[3769,2593,2891,6871],[2163,345,1598,0]])
    return df

def compare_maf_ninjas_entsoe(fig_path='D:/NordicModel/Figures',error_type='mae'):
    #%% compare solar data
    from model_definitions import bidz2maf_pecd
    import maf_pecd_data
    import renewables_ninja_db

    if error_type == 'mae':
        err = lambda x:np.mean(np.abs(x))
    else:
        err = lambda x:np.sqrt(np.mean(np.square(x)))

    starttime = '20160101:00'
    endtime = '20161230:23'
    area = 'DE'

    plot_start = '20160601:00'
    plot_days = 20
    # fig_path = 'D:/NordicModel/Figures'
    normalize = False
    ninja_type = 'current'

    areas = ['DE','GB']


    maf_db = maf_pecd_data.Database(db='D:/Data/maf_pecd.db')
    entsoe_db = Database(db='D:/Data/gen.db')
    ninja_solar_db = renewables_ninja_db.Database('D:/Data/renewables_ninja.db')
    ninja_wind_db = renewables_ninja_db.Database('D:/Data/ninja_wind.db')

    ## ENTSOE DATA
    entsoe_dic = entsoe_db.select_gen_per_type_v2(starttime=starttime,endtime=endtime,areas=areas,types=['Solar','Wind onsh','Wind offsh'])

    ## MAF DATA
    maf_areas = [bidz2maf_pecd[a] for a in areas]
    maf_pv = maf_db.select_pecd_data(starttime=starttime,endtime=endtime,data_type='pv',get_areas=maf_areas)
    maf_pv.columns = areas
    maf_onsh = maf_db.select_pecd_data(starttime,endtime,'onshore',get_areas=maf_areas)
    maf_onsh.columns = areas
    maf_offsh = maf_db.select_pecd_data(starttime,endtime,'offshore',get_areas=maf_areas)
    maf_offsh.columns = areas

    ## NINJA DATA
    ninja_pv = ninja_solar_db.select_data(table='solar_merra2',starttime=starttime,endtime=endtime,countries=areas)
    ninja_onsh = ninja_wind_db.select_wind_data(starttime,endtime,areas,ninja_type,'onshore')
    ninja_offsh = ninja_wind_db.select_wind_data(starttime,endtime,areas,ninja_type,'offshore')

    # always normalize entsoe data
    for a in entsoe_dic:
        entsoe_dic[a] = entsoe_dic[a] / entsoe_dic[a].max()
    if normalize:
        maf_pv = maf_pv / maf_pv.max()
        maf_onsh = maf_onsh / maf_onsh.max()
        maf_offsh = maf_offsh / maf_offsh.max()
        ninja_pv = ninja_pv / ninja_pv.max()
        ninja_onsh = ninja_onsh / ninja_onsh.max()
        ninja_offsh = ninja_offsh / ninja_offsh.max()

    columns = pd.MultiIndex.from_product([['Solar','Onshore','Offshore'],areas],names=['Type','Country'])
    rmse = pd.DataFrame(index=['MAF','Ninja'],columns=columns)

    #%%
    for a in areas:
        rmse.at['MAF',('Solar',a)] = err(maf_pv[area]-entsoe_dic[a]['Solar'])
        rmse.at['Ninja',('Solar',a)] = err(ninja_pv[area]-entsoe_dic[a]['Solar'])
        rmse.at['MAF',('Onshore',a)] = err(maf_onsh[area]-entsoe_dic[a]['Wind onsh'])
        rmse.at['Ninja',('Onshore',a)] = err(ninja_onsh[area]-entsoe_dic[a]['Wind onsh'])
        rmse.at['MAF',('Offshore',a)] = err(maf_offsh[area]-entsoe_dic[a]['Wind offsh'])
        rmse.at['Ninja',('Offshore',a)] = err(ninja_offsh[area]-entsoe_dic[a]['Wind offsh'])

    #%%
    with open(Path(fig_path)/f'wind_pv_validate.txt','wt') as f:
        rmse.to_latex(f,float_format="{:0.3f}".format)
    #%%
    tidxs = pd.date_range(start=str_to_date(plot_start),end=str_to_date(plot_start)+datetime.timedelta(days=plot_days),freq='H')

    f,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    f.set_size_inches(5,10)
    for area in areas:

        ax1.cla()
        ax2.cla()
        ax3.cla()

        maf_pv.loc[tidxs,area].plot(ax=ax1,label='MAF')
        ninja_pv.loc[tidxs,area].plot(ax=ax1,label='Ninja',linestyle='--')
        entsoe_dic[a].loc[tidxs,'Solar'].plot(ax=ax1,label='ENTSO-E',linewidth=0.7,color='k')
        ax1.grid()
        ax1.legend()
        ax1.set_ylabel('Solar')

        maf_onsh.loc[tidxs,area].plot(ax=ax2,label='MAF')
        ninja_onsh.loc[tidxs,area].plot(ax=ax2,label='Ninja',linestyle='--')
        entsoe_dic[a].loc[tidxs,'Wind onsh'].plot(ax=ax2,label='ENTSO-E',linewidth=0.7,color='k')
        ax2.grid()
        ax2.legend()
        ax2.set_ylabel('Onshore wind')

        maf_offsh.loc[tidxs,area].plot(ax=ax3,label='MAF')
        ninja_offsh.loc[tidxs,area].plot(ax=ax3,label='Ninja',linestyle='--')
        entsoe_dic[a].loc[tidxs,'Wind offsh'].plot(ax=ax3,label='ENTSO-E',linewidth=0.7,color='k')
        ax3.legend()
        ax3.grid()
        ax3.set_ylabel('Offshore wind')

        plt.savefig(Path(fig_path) / f'pv_wind_validate_{area}.png')
        plt.savefig(Path(fig_path) / f'pv_wind_validate_{area}.eps')

def merge_exchange_databases(db1='D:/Data/exchange.db',db2='D:/Data/exchange3.db'):
    """
    Merge values from db2 into db1
    Creates new tables and inserts missing values into db1
    :param db1:
    :param db2:
    :return:
    """

    db1 = Database(db=db1)
    db2 = Database(db=db2)

    conn1 = sqlite3.connect(db1.db)
    c1 = conn1.cursor()
    conn2 = sqlite3.connect(db2.db)
    c2 = conn2.cursor()

    tabls1 = get_tables(c1)
    tabls2 = get_tables(c2)

    #%% check which new tables need to be added
    add_tables = [t for t in tabls2 if t.split('_')[0] in ['flow','capacity'] and t not in tabls1]

    #%% check which tables needs to be complemented with new values
    tab_intersect = [t for t in tabls1 if t in tabls2]
    df = pd.DataFrame(index=tab_intersect,columns=['db1','db2'])
    for table in tab_intersect:
        df.at[table,'db1'] = count_values(table,c1)
        df.at[table,'db2'] = count_values(table,c2)

    edit_tables = list(df.index[df['db1'] < df['db2']])

    count_values('connections_flow',c1)
    #%% add new tables
    for tname in add_tables:

        print(f'Adding new table {tname}')
        # create new table in db1
        cmd = f'CREATE TABLE {tname} (time TEXT NOT NULL, value REAL NOT NULL)'
        _execute_(c1,cmd)

        # get all values from db2
        cmd = f'SELECT * FROM {tname}'
        _execute_(c2,cmd)

        for r in c2.fetchall():
            cmd = f"INSERT INTO {tname} (time,value) values('{r[0]}',{r[1]})"
            _execute_(c1,cmd)

        conn1.commit()
    #%% add new values in old tables
    for tname in edit_tables:

        # get current time values
        cmd = f"SELECT time FROM {tname}"
        _execute_(c1,cmd)
        texist = [r[0] for r in c1.fetchall()]

        cmd = f"SELECT time,value FROM {tname}"
        _execute_(c2,cmd)
        count = 0
        for r in c2.fetchall():
            if r[0] not in texist:
                cmd = f"INSERT INTO {tname} (time,value) values('{r[0]}',{r[1]})"
                _execute_(c1,cmd)
                count += 1
        conn1.commit()
        print(f'Added {count} values to {tname}')

if __name__ == "__main__":
    pass

    pd.set_option('display.max_rows',100)
    pd.set_option('display.max_columns',None)


    #%%
    db = Database('D:/Data/capacity.db')
    # areas = ['SE1','FI','DE']
    from model_definitions import all_areas as areas
    year = 2020
    # def get_entsoe_capacity(year=2020,areas=['FI','DI'],db='D:/Data/entsoe_gen_capacity.db'):
    # db = entsoe_transparency_db.Database(db=db)

    df = db.select_capacity_wrap(year,areas)