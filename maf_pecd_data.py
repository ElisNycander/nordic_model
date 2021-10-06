"""
Create database with wind and pv data from ENTSO-E MAF 2020

"""

import numpy as np
import pandas as pd
import datetime
import sqlite3
from pathlib import Path


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



class Database():

    def __init__(self,db='D:/Data/entsoe_maf_pecd.db',maf_path='D:/Data/MAF/Solar and Wind Data'):

        self.db = db
        self.maf_path = maf_path

    def get_pecd_data(self,tab_type='onshore2025',
                      areas=['SE01','SE02','SE03','SE04','DKW1','DKE1','FI00','NON1','NOM1','NOS0'],max_rows = np.inf):

        # maf_db = Database(db='D:/Data/entsoe_maf_pecd.db')

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()


        data_files = {
            'onshore2025':'PECD_2025_Onshore.xlsx',
            'onshore2030':'PECD_2030_Onshore.xlsx',
            'offshore2025':'PECD_2025_Offshore.xlsx',
            'offshore2030':'PECD_2030_Offshore.xlsx',
            'pv2025':'PECD_2025_PV.xlsx',
            'pv2030':'PECD_2030_PV.xlsx'
        }

        # tab_type = 'onshore2025'
        fname = data_files[tab_type]
        #%%
        from sxl import Workbook

        path = Path(self.maf_path)
        wb = Workbook(path / fname)

        # areas = ['SE01']

        timefmt = '%Y%m%d:%H'

        created_tables = []
        for a in areas:
            # data = []
            print(f'Reading data for {a}')
            ws = wb.sheets[a]
            start = False

            ridx = 0
            for row in ws.rows:
                if start:
                    dm = row[0].split('.')
                    day = int(dm[0])
                    month = int(dm[1])
                    hour = int(row[1]) - 1
                    for y,val in zip(years,row[2:]):
                        timestr = datetime.datetime(y,month,day,hour).strftime(timefmt)
                        # data.append((datetime.datetime(int(y),month,day,hour),val))
                        tab_name = f'{tab_type}_{a}_{y}'
                        if tab_name not in created_tables:
                            _create_table_(cursor,name=tab_name,table_type='pecd')
                            created_tables.append(tab_name)
                        cmd = f'INSERT INTO {tab_name} (time,value) VALUES ("{timestr}",{val})'
                        _execute_(cursor,cmd)
                else:
                    if row[0] == 'Date':
                        # get years
                        years = [int(y) for y in row[2:]]
                        start = True

                if ridx > max_rows:
                    break
                ridx += 1

            conn.commit()
        conn.close()

    def select_pecd_data(self,starttime='20150101:00',endtime='20151501:00',data_type='onshore',model_year=2025,get_areas=['SE01','SE02']):

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        #%% select data
        # data_type = 'onshore'

        # model_year = 2025

        tab_type = f'{data_type}{model_year}'

        # get_areas = ['SE01','SE02']

        strfmt = '%Y%m%d:%H'
        # starttime = '20150101:00'
        # endtime = '20150215:00'

        tab_years = [f'{y}' for y in range(int(starttime[:4]),int(endtime[:4])+1)]


        #%%
        cmd = "SELECT name FROM sqlite_master WHERE type='table';"
        _execute_(cursor,cmd)

        rel_tables = [r[0] for r in cursor.fetchall() if tab_type in r[0] \
                      and r[0].split('_')[1] in get_areas and r[0].split('_')[2] in tab_years]


        # create dataframe
        df = pd.DataFrame(dtype=float,columns=get_areas,index=pd.date_range(
            start=datetime.datetime.strptime(starttime,strfmt),
            end=datetime.datetime.strptime(endtime,strfmt),
            freq='H'))

        for tab in rel_tables:
            area = tab.split('_')[1]
            cmd = f'SELECT time,value FROM {tab} WHERE time >= "{starttime}" AND time <= "{endtime}"'
            _execute_(cursor,cmd)

            for r in cursor.fetchall():
                time = datetime.datetime.strptime(r[0],strfmt)
                df.at[time,area] = r[1]

        conn.close()
        return df

if __name__ == "__main__":

    def main():
        pass

    pass


    #%%
    # inflow_se = inflow_energiforetagen_iso()

    maf_db = Database(db='D:/Data/entsoe_maf_pecd.db')

    df = maf_db.select_pecd_data('20151201:00','20160115:00')

