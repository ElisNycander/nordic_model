import pandas as pd

MWtoGW = 1e-3
GWtoMW = 1e3
GWtoTW = 1e-3
TWtoGW = 1e3
cm_per_inch = 2.5
std_fig_size = (6.4,4.8)

# rough EU ETS CO2 price
co2_price_ets = { # EUR/ton C02
    2016:6,
    2017:7,
    2018:15,
    2019:25,
    2020:25,
    2021:50,
}

solver_stats = { # solver stats (different names for different apis)
    'pyomo':{
        'NumVars':'Number of variables',
        'NumConstrs':'Number of constraints',
        'NumNZs':'Number of nonzeros',
    },
    'gurobi':{
        'NumVars':'NumVars',
        'NumConstrs':'NumConstrs',
        'NumNZs':'NumNZs'
    },
}
solver_executables = { # specify paths to solver executables
    # 'ipopt':'C:\\ipopt-3.14\\bin\\ipopt',
    'ipopt':'C:\\ipopt\\bin\\ipopt',
}

# mapping from bid zones to maf data for wind and solar
bidz2maf_pecd = {
    'SE1':'SE01',
    'SE2':'SE02',
    'SE3':'SE03',
    'SE4':'SE04',
    'FI':'FI00',
    'NO1':'NOS0',
    'NO2':'NOS0',
    'NO5':'NOM1',
    'NO3':'NOM1',
    'NO4':'NON1',
    'DK1':'DKW1',
    'DK2':'DKE1',
    'EE':'EE00',
    'LT':'LT00',
    'LV':'LV00',
    'DE':'DE00',
    'PL':'PL00',
    'GB':'UK00',
    'NL':'NL00',
}

# Note: connections between internal and external areas MUST be entered with the
# external area in the 'to' columns: internal->external
nordpool_capacities = pd.DataFrame(columns=['from','to','c1','c2'],data = [
    ['SE1','SE2',3300,3300],
    ['SE2','SE3',7300,7300],
    ['SE3','SE4',5400,2000],
    ['SE1','FI',1600,1150],
    ['SE3','FI',1200,1200],
    ['SE1','NO4',600,700],
    ['SE2','NO4',300,250],
    ['SE3','NO1',2095,2145],
    ['SE2','NO3',1000,600],
    ['SE3','DK1',680,740],
    ['SE4','DK2',1300,1700],
    ['SE4','LT',700,700],
    ['SE4','PL',600,600],
    ['SE4','DE',615,615],
    ['FI','EE',1016,1016],
    ['EE','LV',1000,879],
    ['LV','LT',1302,684],
    ['NO4','NO3',1200,500],
    ['NO3','NO5',500,700],
    ['NO3','NO1',500,500],
    ['NO5','NO1',3900,600],
    ['NO5','NO2',600,500],
    ['NO1','NO2',2200,3500],
    ['DK1','DK2',590,600],
    ['NO2','NL',723,723],
    ['DK1','DE',2500,2500],
    ['NO2','DK1',1632,1632],
    ['DK2','DE',773,1000],
    ['LT','PL',500,500],
    ['PL','DE',2000,2000],
    ['DE','NL',4000,4000],
    ['NL','GB',1000,1000],
    ['NL','BE',2000,2000], # external connections start here
    ['GB','IE',1000,1000], # Note: If fixed transfer option is used, the capacity to external areas is not used
    ['GB','FR',3000,3000], # but the connection must be entered here for the model to use data for the connection
    ['EE','RU',0,0],
    ['LV','RU',320,320],
    ['FI','RU',320,1300],
    ['LT','BY',400,400],
    ['DE','FR',0,0], # new connections from here
    ['DE','CZ',0,0],
    ['DE','AT',0,0],
    ['DE','CH',0,0],
    ['PL','CZ',0,0],
    ['PL','UA',0,0],
    ['PL','SK',0,0]
])

new_trans_cap_columns = ['year','from','to','c1_change','c2_change','c1','c2','name/info']
# new transmission capacities
new_trans_cap = pd.DataFrame([
    [2020,'NO2','DE',1400,1400,1400,1400,'NordLink'],
    [2020,'DK1','NL',700,700,700,700,'COBRAcable'],
    [2021,'SE3','SE4',600,600,6600,3200,'South-West Link'],
    [2021,'NO2','GB',1400,1400,1400,1400,'North Sea Link'],
    [2021,'DK1','DE',720,1000,2500,2500,'Stage 1 Jylland-Germany'],
    [2022,'SE2','SE3',500,500,7800,7800,'Reinforcement snitt 2'],
    [2023,'SE2','SE3',300,300,8100,8100,'Reinforcement snitt 2'],
    [2023,'SE3','SE4',600,400,7200,3600,'Ekhyddan-Nybro-Hemsjö'],
    [2023,'DK1','DE',1000,1000,3500,3500,'Stage 2 Jylland-Germany'],
    [2023,'DK1','GB',1400,1400,1400,1400,'Viking Link'],
    [2026,'SE4','DE',700,700,1315,1315,'Hansa PowerBridge'],
    [2026,'SE1','FI',800,900,3200,3200,'3:rd AC'],
    [2029,'SE3','FI',-400,-400,800,800,'Decommisioning Fenno-Skan1'],
    [2029,'SE2','FI',800,800,800,800,'HVDC FI-SE2'],
    [2035,'NO2','GB',1400,1400,2800,2800,'NorthConnect'], # not sure about NO2
    [2035,'SE2','SE3',2400,2400,10500,10500],
],columns=new_trans_cap_columns)

generators_def = {
    'SE1':['Thermal','Hydro'],
    'SE2':['Thermal','Hydro'],
    'SE3':['Nuclear','Thermal','Hydro'],
    'SE4':['Thermal','Hydro'],
    'NO1':['Hydro','Thermal'], # from entso-e data it seems only NO4 and NO5
    'NO2':['Hydro'], # have significant thermal generation
    'NO3':['Hydro'],
    'NO4':['Hydro','Thermal'],
    'NO5':['Hydro','Thermal'],
    'DK1':['Thermal'],
    'DK2':['Thermal'],
    'FI':['Nuclear','Thermal','Hydro'],
    'LV':['Thermal','Hydro'],
    'LT':['Thermal','Hydro'],
    'EE':['Thermal'],
    'PL':['Thermal','Hydro'],
    'GB':['Nuclear','Thermal','Hydro'],
    'DE':['Nuclear','Thermal','Hydro'],
    'NL':['Nuclear','Thermal'],
}

# Mapping of entsoe production categories to production types used in model
# Used when fitting cost coefficients, and calculating generation capacity limits
entsoe_type_map = {
    'Thermal':['Biomass','Brown coal','Coal-gas','Gas','Hard coal','Oil','Oil shale','Peat','Waste','Other'],
    'Hydro':['Hydro ror','Hydro res','Hydro pump'],
    'Wind':['Wind offsh','Wind onsh'],
    'Nuclear':['Nuclear'],
}

colors = {'Hydro':'skyblue',
          'Slow':'#ff7f0e',
          'Fast':'#d62728',
          'Nuclear':'#8c564b',
          'Wind':'#2ca02c',
          'Thermal':'#ff7f0e',
          'Solar':'khaki',
          'HROR':'darkorchid',
          'PUMP':'teal',
          'REL':'teal',
}

country_to_areas = {
    'DK':['DK1','DK2'],
    'FI':['FI'],
    'NO':['NO1','NO2','NO3','NO4','NO5'],
    'SE':['SE1','SE2','SE3','SE4'],
    'LT':['LT',],
    'LV':['LV',],
    'EE':['EE',],
    'DE':['DE',],
    'GB':['GB',],
    'NL':['NL',],
    'PL':['PL',],
}

area_to_country = {}
for c in country_to_areas:
    for a in country_to_areas[c]:
        area_to_country[a] = c

all_areas = ['SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2','LV','LT','EE','GB','DE','NL','PL']

all_areas_no_SE = ['NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2',
                   'EE','LT','LV','GB','PL','NL','DE']

# all areas inside nordic synchronous region
synchronous_areas = ['SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5','FI','DK2']

nordpool_areas = ['SE1','SE2','SE3','SE4','NO1','NO2','NO3','NO4','NO5','FI','DK1','DK2','EE','LT','LV']


all_countries = list(set([area_to_country[a] for a in all_areas]))