
import csv
import sqlite3
import pandas as pd

from help_functions import create_select_list

code2opsd = {
    'SE':'SE',
    'NO':'NO',
    'FI':'FI',
    'DK':'DK',
    'PL':'PL',
    'NL':'NL',
    'FR':'FR',
    'ES':'ES',
    'GB':'UK',
    'IT':'IT',
    'CH':'CH',
    'AT':'AT',
    'CZ':'CZ',
    'SK':'SK',
    'SI':'SI',
}

code2res = {
    'SE':'Sweden',
    'NO':'Norway',
    'DK':'Denmark',
    'FI':'Finland',
    'EE':'Estonia',
    'LT':'Lithuania',
    'LV':'Latvia',
    'GB':'United Kingdom',
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
    'CZ':'Czech Republic',
    'SK':'Slovakia',
    'SI':'Slovenia',
    'HU':'Hungary',
    'PL':'Poland',
    'CR':'Croatia',
    'SR':'Serbia',
    'BL':'Bulgaria',
    'RO':'Romania',
    'MT':'Montenegro',
    'MK':'The former Yugoslav Republic of Macedonia',
    'GR':'Greece',
    'AL':'Albania',
    'BH':'Bosnia and Herzegovina',
}

code2wri = {
    'SE':'SWE',
    'NO':'NOR',
    'FI':'FIN',
    'DK':'DNK',
    'EE':'EST',
    'LT':'LTU',
    'LV':'LVA',
    'PL':'POL',
    'NL':'NLD',
    'FR':'FRA',
    'BE':'BEL',
    'DE':'DEU',
    'ES':'ESP',
    'PT':'PRT',
    'IE':'IRL',
    'GB':'GRB',
    'IT':'ITA',
    'CH':'CHE',
    'AT':'AUT',
    'CZ':'CZE',
    'SK':'SVK',
    'HU':'HUN',
    'SI':'SVN',
    'CR':'HRV',
    'BL':'BGR',
    'BH':'BHI',
    'MK':'MKD',
    'SR':'SRB',
    'GR':'GRC',
    'RO':'ROU',
    'MT':'MNE',
    'AL':'ALB',
}

class Database():

    def __init__(self, db='C:/Data/power_plants.db'):

        self.db = db

    def create_table(self,name,cols,not_null=None):
        """ Create table with name "name", and columns given by the
        keys of col_dict, with the fields specifying the type of variable


        cols = {
            col_name:col_type, TEXT/REAL
        }
        # which columns should not be empty
        none_cols = {
            col_name:False
        }

        """
        if not_null is None:
            not_null = {}
            for c in cols:
                not_null[c] = False

        # make sqlite database
        conn = sqlite3.connect(self.db)
        # print(sqlite3.version)
        c = conn.cursor()

        c.execute(f'DROP TABLE IF EXISTS {name}')

        cr_tb_str = f'CREATE TABLE {name} ('

        for col in cols:
            if not_null[col]:
                cr_tb_str += f'{col} {cols[col]} NOT NULL,'
            else:
                cr_tb_str += f'{col} {cols[col]},'
                
        # strip last comma
        cr_tb_str = cr_tb_str.rstrip(',')
        cr_tb_str += ')'
        
        c.execute(cr_tb_str)
        conn.commit()
        conn.close()
        

    def create_wri_table(self):

        WRI_file = 'C:/Data/globalpowerplantdatabasev120/global_power_plant_database.csv'
        tab_name = 'WRI'

        with open(WRI_file, encoding='utf-8') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')
            ridx = 0
            for row in reader:
                if ridx == 0:
                    # find fields
                    fields = row
                    fields_str = ''
                    for f in fields:
                        fields_str += f'{f},'
                    fields_str = fields_str.rstrip(',')
                else:
                    if ridx == 1:
                        # use first data row to check which data is numeric
                        # create dict with fields
                        tab_cols = {}
                        for idx, val in enumerate(row):
                            try:
                                float(val)
                                tab_cols[fields[idx]] = 'REAL'
                            except ValueError:
                                tab_cols[fields[idx]] = 'TEXT'
                        self.create_table(name=tab_name, cols=tab_cols)

                        # open database
                        conn = sqlite3.connect(self.db)
                        c = conn.cursor()

                    # enter data into database

                    val_str = ''
                    for idx, val in enumerate(row):
                        if val == '':
                            val_str += 'NULL,'
                        else:
                            if tab_cols[fields[idx]] == 'REAL':
                                val_str += f'{val},'
                            else:
                                # remove "" to prevent error
                                val_1 = val.replace('"', '')
                                val_str += f'"{val_1}",'
                    val_str = val_str.rstrip(',')

                    insert_cmd_str = f'INSERT INTO {tab_name} ({fields_str}) values ({val_str})'
                    c.execute(insert_cmd_str)
                ridx += 1
            conn.commit()
            conn.close()

    def create_reservoir_table(self):

        file = 'C:/Data/Europe-dams_eng.xlsx'
        table = 'reservoirs'
        import xlrd
        wb = xlrd.open_workbook(file)
        sheet = wb.sheet_by_name('Dams')

        title_row = 1
        ncols = sheet.ncols
        cols = []
        for i in range(ncols):
            cols.append(sheet.cell(title_row, i).value)

        fcols = [
            c.replace('\n', ' ').replace('-', '').replace('/', '').replace(' ', '_').replace('(', ' ').replace(')', ' ')
            for c in cols]
        fcols = [c.split(' ')[0].strip('_') for c in fcols]

        # numeric columns
        numeric_idx = [9, 10, 11, 12, 13, 23, 24]
        # for i in numeric_idx:
        #     print(fcols[i])

        not_null_list = ['Name_of_dam']
        # %%
        tab_cols = {}
        for cidx, c in enumerate(fcols):
            if cidx in numeric_idx:
                tab_cols[c] = 'REAL'
            else:
                tab_cols[c] = 'TEXT'
        not_null = {}
        for c in fcols:
            if c in not_null_list:
                not_null[c] = True
            else:
                not_null[c] = False

        # create table
        self.create_table(table, cols=tab_cols, not_null=not_null)

        # %% enter data in table
        col_str = ''
        for c in fcols:
            col_str += f'{c},'
        col_str = col_str.strip(',')

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        for ridx in range(2, sheet.nrows):

            val_str = ''
            for cidx in range(sheet.ncols):
                val = sheet.cell(ridx, cidx).value
                # format value to avoid errors
                if type(val) is str:
                    val = val.replace('"', '')
                    if tab_cols[fcols[cidx]] == 'REAL':
                        val = 'NULL'
                    if fcols[cidx] == 'Name_of_dam' and val == '':
                        val = 'Unknown'

                if val == '':
                    val_str += "NULL,"
                elif tab_cols[fcols[cidx]] == 'TEXT':
                    val_str += f'"{val}",'
                else:
                    val_str += f'{val},'
            val_str = val_str.strip(',')

            # execute insert
            cmd = f"INSERT INTO {table} ({col_str}) values ({val_str})"
            c.execute(cmd)

        conn.commit()
        conn.close()

    def get_table_columns(self,table):

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        # %% get table names
        cmd = f"SELECT sql FROM sqlite_master WHERE tbl_name = '{table}' AND type = 'table'"

        c.execute(cmd)
        for row in c:
            str = row[0]

        # str1 = str.split('(')[1].strip(')')
        # cols = [s.split(' ')[0] for s in str1.split(',')]
        str1 = str.split('(')[1].strip(')').replace('\n','').replace('"','')
        str2 = str1.split(',')
        cols = [s.strip(' ').split(' ')[0] for s in str2]

        conn.close()
        return cols

    def select_data(self,table='WRI',select_column='country',column_vals=['SWE']):

        tab_cols = self.get_table_columns(table)
        sel_vals = create_select_list(column_vals)

        # open database
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        # %% build commands

        count_cmd = f'SELECT count(*) FROM {table} '
        select_cmd = f'SELECT * FROM {table} '
        if select_column is not None:
            select_cmd += f'WHERE {select_column} IN {sel_vals}'
            count_cmd += f'WHERE {select_column} IN {sel_vals}'

        # %% get number of rows
        c.execute(count_cmd)
        for row in c:
            nrows = row[0]

        # %% create dataframe
        df = pd.DataFrame(columns=tab_cols, index=range(nrows))

        # %% put data into dataframe
        c.execute(select_cmd)
        ridx = 0
        for row in c:
            for idx, val in enumerate(row):
                df.iat[ridx, idx] = val
            ridx += 1

        # %% drop columns with all None
        drop_cols = df.columns[df.isna().sum() == nrows]
        df = df.drop(drop_cols, axis=1)

        conn.close()

        # TODO: convert dtypes
        return df


def create_aggregate_capacity_excel():
    """ Create excel file with aggregate hydro capacities """


    db = Database(db='D:/Data/power_plants.db')
    db_opsd = Database(db='D:/Data/conventional_power_plants.sqlite')

    wri_df = db.select_data(table='WRI',select_column='primary_fuel',column_vals=['Hydro'])
    res = db.select_data(table='reservoirs',select_column=None,column_vals=['Sweden'])

    # df = df.loc[df.primary_fuel=='Hydro',:]
    opsd_df_eu = db_opsd.select_data(table='conventional_power_plants_EU',select_column='energy_source',column_vals=['Hydro'])
    opsd_df_de = db_opsd.select_data(table='conventional_power_plants_DE',select_column='fuel',column_vals=['Hydro'])

    areas = [
        'SE','NO','FI','DK','EE','LT','LV','PL','NL','FR','BE','DE',
        'ES','PT','IE','GB','IT','CH','AT','CZ','CH','SK','HU','SI','CR','BL','BH','MK','SR','GR','RO','MT','AL'
    ]


    #%%
    df = pd.DataFrame(0,index=areas,columns=['name','WRI MW', 'WRI #','OPSD MW','OPSD #','res MCM', 'res #'])
    df = df.astype(dtype={
        'WRI MW':float,'WRI #':int,'OPSD MW':float,'OPSD #':int,'res MCM':float,'res #':int,'name':str
    })

    """
    res cap - reservoir capacity (million m3)

    """
    from entsoe_transparency_db import tbidz_name
    for a in areas:
        df.at[a,'name'] = tbidz_name[a]
        if a in code2wri:
            df_a = wri_df.loc[[i for i in wri_df.index if wri_df.at[i,'country'] == code2wri[a]],:]
            df.at[a,'WRI MW'] = df_a['capacity_mw'].sum()
            df.at[a,'WRI #'] = df_a.__len__()

        if a in code2res:
            res_a = res.loc[[i for i in res.index if res.at[i,'Country'] == code2res[a]],:]
            df.at[a,'res MCM'] = res_a['Reservoir_capacity'].sum()
            df.at[a,'res #'] = res_a.__len__()

        if a in code2opsd:
            df_opsd_a = opsd_df_eu.loc[[i for i in opsd_df_eu.index if opsd_df_eu.at[i,'country'] == code2opsd[a]],:]
            df.at[a,'OPSD MW'] = df_opsd_a['capacity'].sum()
            df.at[a,'OPSD #'] = df_opsd_a.__len__()

    df.at['DE','OPSD MW'] = opsd_df_de['capacity_gross_uba'].sum()
    df.at['DE','OPSD #'] = opsd_df_de.__len__()

    # wri_areas = [code2wri[a] for a in areas if a in code2wri]
    # df = df.loc[[i for i in df.index if df.at[i,'country'] in wri_areas],:]

    df.to_excel('databases_hydro_capacity.xlsx')

if __name__ == "__main__":

    create_aggregate_capacity_excel()


    #%%
