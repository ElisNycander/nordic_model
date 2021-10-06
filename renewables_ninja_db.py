# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:46:32 2019

@author: elisn
"""

from pathlib import Path
import sqlite3
import csv
from help_functions import str_to_date, cet2utc
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

wind_types = {
    'onshore':'on',
    'offshore':'off',
    'national':'nat',
}

def _create_table_(c, name):
    """ Drop table if it exists, then create new table with given name """

    c.execute(f'DROP TABLE IF EXISTS {name}')

    cmd = f'CREATE TABLE {name} (' + \
    'time TEXT NOT NULL,' + \
    'val REAL NOT NULL' + \
    ')'
    _execute_(c,cmd)

def _execute_(c, cmd):
    try:
        c.execute(cmd)
    except sqlite3.Error as e:
        print('Executing command ''{0}'' returned error: {1}'.format(cmd, e))


def _get_tables_(c,cmd):
    """
    Find all tables in database
    :param c:
    :param cmd:
    :return:
    """
    cmd = "SELECT name FROM sqlite_master WHERE type='table'"
    _execute_(c,cmd)
    t = []
    for row in c.fetchall():
        t.append(row[0])
    return t

class Database():
    
    def __init__(self,db='Data/renewables_ninja.db'):
        
        self.db = db
        
        if Path(self.db).exists():
            pass


    def get_wind_merra2(self,filepath='D:/Data/renewables_ninja/',countries=['DE'],
                        data_type='current'):
        """
        Read csv data for wind merra-2 into database, one table per year
        The following data exists: [current,near-termfuture]x[onshore,offshore,national]
        Data types: current/nearterm

        Table names are:
        wind_[country]_[current/nearterm]_[on/off/nat]
        """

        data_type_str = {
            'current':'current',
            'nearterm':'near-termfuture'
        }

        # first scan for files
        files = [f for f in os.listdir(filepath) if os.path.isfile(Path(filepath)/f) and 'wind' in f \
                 and data_type_str[data_type] in f]
        if countries: # select files for given countries
            files = [f for f in files if sum(1 for c in countries if c in f)]

        #%%
        conn = sqlite3.connect(self.db)
        c = conn.cursor()


        # list of created tables
        tables = []
        read_fmt = '%Y-%m-%d %H:%M:%S'
        write_fmt = '%Y%m%d:%H'
        for file in files:
            print(f'Reading {file}')
            country = file.split('_')[3]

            with open(Path(filepath)/file) as csv_file:

                csv_reader = csv.reader(csv_file,delimiter=',')
                for row in csv_reader:
                    # only store non-zero values
                    try:
                        year = int(row[0][0:4]) # first 4 digits are year for non-zero entries
                        datestr = datetime.datetime.strptime(row[0],read_fmt).strftime(write_fmt)
                        for cidx,wtype in enumerate(list(wind_types.keys())):
                            table = f'wind_{country}_{data_type}_{wtype}'
                            if table not in tables:  # create new table
                                _create_table_(c,table)
                                tables.append(table)

                            val = row[1+cidx]
                            cmd = f"INSERT INTO {table} (time,val) VALUES ('{datestr}',{val})"
                            _execute_(c,cmd)

                    except ValueError:
                        pass

            conn.commit()
        conn.close()


    def get_solar_merra2(self,filepath='C:/Users/elisn/Box Sync/Data/renewables_ninja/',
                         countries = ['SE','DK','FI','NO','PL',]):
        """ Read csv data for solar merra-2 into database, one table per year """
        
        # filepath = 'C:/Users/elisn/Box Sync/Data/renewables_ninja/'
        
        # countries = ['SE','DK','FI','NO','PL',]
        
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        
        # drop all tables  with solar_merra2 data
        c.execute("SELECT name FROM sqlite_master WHERE type='table'");
        for t in c.fetchall():
            if 'solar_merra2' in t[0]:
                c.execute('DROP TABLE {0}'.format(t[0]))
            
        # list of created tables
        tables = []
          
        for country in countries:
            file = 'ninja_pv_country_{0}_merra-2_corrected.csv'.format(country)
    
            with open(filepath + file) as csv_file:
                
                csv_reader = csv.reader(csv_file,delimiter=',')
                for row in csv_reader:
                    # only store non-zero values
                    try: 
                        int(row[0][0:4]) # first 4 digits are year for non-zero entries
                        
                        table = 'solar_merra2_{0}'.format(row[0][:4]) # table name includes year
                        if table not in tables:
                            # create new table 
                            c.execute('CREATE TABLE {0} ('.format(table) + \
                                'time TEXT NOT NULL,' + \
                                'country TEXT NOT NULL,' + \
                                'prod REAL' + \
                                ')')
                            tables.append(table)
                            
                        date_str = row[0][0:10].replace('-','')
                        hour_str = row[0][11:13]
                        datetime_str = date_str + ':' + hour_str
                        val = row[1]
                        cmd = "INSERT INTO {3} (time,country,prod) VALUES ('{0}','{1}',{2})".format(datetime_str,country,val,table)
                        #print(cmd)
                        c.execute(cmd)
                    except ValueError:
                        pass
    
        conn.commit()
        conn.close()
        
    def select_data(self,table='solar_merra2',starttime = '20160101:00',endtime = '20160107:23',countries=[],cet_time=False):
        """ Select data from tables
        Implemented tables: 
            solar_merra2
        
        """
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        if cet_time:
            starttime = cet2utc(starttime)
            endtime = cet2utc(endtime)

        if table == 'solar_merra2':
            if int(starttime[:4]) < 1980 or int(endtime[:4]) > 2019:
                print('Given time outside available range, 1985-2016')
                return None
        else:
            print("Unknown table '{0}'".format(table))
            return None
        
        tables = ['solar_merra2_{0}'.format(year) for year in range(int(starttime[:4]),int(endtime[:4])+1)]
    
        if not countries == []:
            columns = countries
            str_countries = "'{0}'".format(countries[0])
            for country in countries[1:]:
                str_countries += ",'{0}'".format(country) 
        else:
            # get list of countries
            c.execute("SELECT DISTINCT country FROM solar_merra2_1985")
            columns = []
            for country in c.fetchall():
                columns.append(country[0])
                
        data = pd.DataFrame(0,dtype=float,index=pd.date_range(str_to_date(starttime),str_to_date(endtime),freq='H'),columns=columns)
        
        
        for t in tables:
            
            cmd = "SELECT time,country,prod FROM {0} WHERE time >= '{1}' AND time <= '{2}'".format(t,starttime,endtime)
            if not countries == []:
                cmd += ' AND country IN ({0})'.format(str_countries)
            #print(cmd)
                
            
            c.execute(cmd)
            for row in c.fetchall():
                #print(row)
                data.at[str_to_date(row[0]),row[1]] = row[2]

        if cet_time:
            data.index = data.index + datetime.timedelta(hours=1)

        return data

    def select_wind_data(self,starttime='20160101:00',endtime='20160107:23',countries=['DE'],
                         data_type='current',wind_type='onshore',cet_time=False):
        """
        :param data_type: current/nearterm
        :param wind_type: onshore/offshore/national
        :return:
        """

        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        if cet_time:
            starttime = cet2utc(starttime)
            endtime = cet2utc(endtime)

        # get relevant tables
        rel_tables = [t for t in _get_tables_(c,conn) if 'wind' in t and data_type in t and wind_type in t \
                      and sum(1 for c in countries if c in t)]

        df = pd.DataFrame(dtype=float,index=pd.date_range(str_to_date(starttime),str_to_date(endtime),freq='H'),columns=countries)
        for table in rel_tables:
            country = table.split('_')[1]
            cmd = f"SELECT time,val from {table} WHERE time >= '{starttime}' and time <= '{endtime}'"
            _execute_(c,cmd)
            for row in c.fetchall():
                df.at[str_to_date(row[0]),country] = row[1]

        if cet_time:
            df.index = df.index + datetime.timedelta(hours=1)
        return df

def compare_solar_generation():
    """ Compare solar generation from ENTSO-E data with solar generation from Merra data (using 
    installed capacities) """
    
       
    db = Database()
 
    starttime = '20140101:00'
    endtime = '20141207:23'
    
    
    data_cf = db.select_data(starttime=starttime,endtime=endtime)
    
    # get solar data from ENTSO-E data
    
    # installed solar capacity (MW)
    pv_capacities = { # 2017, from IRENA RE Capacity
            'DK':900, # from Energy Outlook 2018
            'FI':50,
            'NO':42,
            'SE':254,
            }
    data = pd.DataFrame(index=data_cf.index,columns=data_cf.columns)
    for col in data.columns:
        data.loc[:,col] = data_cf.loc[:,col] * pv_capacities[col]
    
    import entsoe_transparency_db as entsoe
    entsoe_db = entsoe.Database()
    
    areas = ['SE1','SE2','SE3','SE4','DK1','DK2']
    edata = entsoe_db.select_gen_per_type_wrap(starttime='20180101:00',endtime='20181207:23',areas=areas)
    for cc in edata:
        edata[cc].index = data.index
    
    esolar = pd.DataFrame(0,index=edata['SE1'].index,columns=['SE','DK'],dtype=float)
    for searea in ['SE1','SE2','SE3','SE4']:
        esolar.loc[:,'SE'] += edata[searea].loc[:,'Solar']
    for dkarea in ['DK1','DK2']:
        esolar.loc[:,'DK'] += edata[dkarea].loc[:,'Solar']
  
    
    ax = data['SE'].plot()
    esolar['SE'].plot(ax=ax)
    plt.title('SE')
    plt.legend(['Merra','ENTSO-E'])
    plt.grid()
    plt.show()
    
    ax = data['DK'].plot()
    esolar['DK'].plot(ax=ax)
    plt.title('DK')
    plt.legend(['Merra','ENTSO-E'])    
    plt.grid()
    plt.show()
        
        
if __name__ == '__main__':
    
    #compare_solar_generation()
    
    
    
    db = Database(db='D:/Data/ninja_wind.db')
    # for data_type in ['current','nearterm']:
    #     db.get_wind_merra2(countries=['DE','GB','FI'],data_type=data_type)

    # db.get_solar_merra2(filepath='D:/Data/renewables_ninja/',countries=['SE','NO','FI','DK','PL','DE','NL','GB','LT','LV','EE'])
    # db.get_solar_merra2(filepath='C:/Users/elisn/Box Sync/Data/renewables_ninja/')
    
    # df = db.select_data(table='solar_merra2',starttime='20150101:00',endtime='20191231:23',countries=['SE'])

    # print excel file
    # df.to_excel('PV_SE_kapacitetsfaktor.xlsx')

    #%% select wind data

    data_type='current' # current/nearterm
    wind_type='national' # onshore/offshore/national

    starttime = '20160101:00'
    endtime = '20160107:23'
    countries=['DE','GB']
    cet_time=False


    df = db.select_wind_data(starttime,endtime,countries,data_type,wind_type)
    df.plot()