
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

from model_definitions import MWtoGW, GWtoTW
from help_functions import str_to_date

class GurobiModel:

    def __init__(self,name):

        self.gm = gp.Model(name)
        self.name = name
        self.api = 'gurobi'

    def setup_opt_problem(self,m,highres=False,delta=None):
        """
        Setup optimization, assumes all necessary data structures have been created
        Note: m is the owner of the PyomoModel, which contains all options and
        data-structures needed to setup the optimization problem

        highres:True - setup 15min version of the model
                False - either original hourly model (if delta=None) or model with resolution delta>=1

        # naming conventions:
        set_XXX - set
        par_XXX - parameter
        var_XXX - variable
        """
        # Set parent model "pm"
        self.pm = m
        prt = m.opt_print['setup']

        ## SETS

        self.set_TIME = m.idx_time
        self.set_DAYS = m.idx_day
        self.set_AREA = m.areas
        self.set_COUNTRY = m.opt_countries
        self.set_C2A = m.country_to_areas
        self.set_SYNCAREA = m.syncareas
        self.set_ROR_AREA = m.ror_areas
        self.set_PUMP_AREA = m.pump_areas
        self.set_ROR_COUNTRY = m.ror_countries
        self.set_PUMP_RES_AREA = m.pump_res_areas
        self.set_PUMP_NORES_AREA = m.pump_nores_areas

        # all gens
        self.set_GEN = m.idx_gen
        # hydro gens
        self.set_HYDRO_GEN = [i for i in m.idx_gen if m.gen_data.at[i,'gtype'] == 'Hydro']
        # all gens - hydro gens
        self.set_THERMAL_GEN = [i for i in m.idx_gen if not m.gen_data.at[i,'gtype'] == 'Hydro']
        # nuclear
        self.set_NUCLEAR_GEN = [i for i in m.idx_gen if m.gen_data.at[i,'gtype'] == 'Nuclear']
        # define set of combined generators
        self.set_COMB_GEN = range(1,m.nGenComb+1)
        self.set_COMB_GEN_TO_GEN = initialize = m.gen_comb
        self.set_RESERVE_AREA = m.resareas
        self.set_RESERVE_AREA_TO_GEN = m.reserve_gens
        self.set_RESERVE_COUNTRY = [c for c in m.opt_reserves_fcrn if m.opt_reserves_fcrn[c] > 0]
        self.set_RESERVE_COUNTRY_TO_GEN = m.reserve_gens_country

        self.set_WIND_AREA = m.wind_areas
        self.set_AREA_TO_GEN = m.gen_in_area

        self.set_HYDRO_AREA = m.hydrores
        self.set_HYDRO_AREA_TO_GEN = m.reservoir2hydro
        # all areas with positive solar capacity
        self.set_SOLAR_AREA = m.solar_areas

        # internal connections
        self.set_XINT = [i for i in range(1,m.nXint+1)]
        self.set_XINT_FW = m.xintf
        self.set_XINT_BW = m.xintr

        # external connections
        self.set_XEXT = [i for i in range(1,m.nXext+1)]
        # divide external connections into fixed price and fixed transfer
        # fixed price connections
        self.set_XEXT_VAR = m.fixed_price_connections
        # fixed transfer connections
        self.set_XEXT_PAR = m.fixed_transfer_connections
        self.set_AREA_TO_XEXT_PAR = m.xext_ft
        self.set_AREA_TO_XEXT_VAR = m.xext_fp

        self.set_HVDC = [i for i in m.combined_hvdc.keys()]
        self.set_HVDC_TO_XINT = m.combined_hvdc

        # define the sets over which the variables are defined
        self.var_sets = {
            # 'D':['AREA','TIME'],
            'LS':['AREA','TIME'],
            'PG':['GEN','TIME'],
            'WIND':['WIND_AREA','TIME'],
            'SOLAR':['SOLAR_AREA','TIME'],
            'HROR':['ROR_AREA','TIME'],
            'X1':['XINT','TIME'],
            'X2':['XINT','TIME'],
            'XEXT':['XEXT_VAR','TIME'],
            'XMOD':['XEXT_MOD','TIME'],
            'PUMP':['PUMP_AREA','TIME'],
            'REL':['PUMP_RES_AREA','TIME'],
            'PRES':['PUMP_RES_AREA','TIME'],
        }
        if m.opt_hydro_daily:
            self.var_sets['RES'] = ['HYDRO_AREA','DAYS']
            self.var_sets['SPILLAGE'] = ['HYDRO_AREA','DAYS']
        else:
            self.var_sets['RES'] = ['HYDRO_AREA','TIME']
            self.var_sets['SPILLAGE'] = ['HYDRO_AREA','TIME']

        # choose variable limits, different for hourly and high resolution model
        max_val = {}
        for var in ['LS','PG','WIND','SOLAR','HROR','X1','X2','XEXT','RES','PUMP','PRES']:
            max_val[var] = m.__getattribute__(f'max_{var}')

        min_val = {}
        for var in ['PG','XEXT']:
            min_val[var] = m.__getattribute__(f'min_{var}')

        self.max_val = max_val
        self.min_val = min_val

        ## VARIABLES
        if prt:
            print('Setting up VARIABLES')
        # curtailable demand
        self.var_LS = self.gm.addVars(self.set_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,name='LS',lb=0,ub=max_val['LS'])

        ## all generators
        # production
        self.var_PG = self.gm.addVars(self.set_GEN,self.set_TIME,vtype=GRB.CONTINUOUS,name='PG',lb=min_val['PG'],ub=max_val['PG'])
        # wind generation
        self.var_WIND = self.gm.addVars(self.set_WIND_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,name='WIND',lb=0,ub=max_val['WIND'])
        # solar generation
        self.var_SOLAR = self.gm.addVars(self.set_SOLAR_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,name='SOLAR',lb=0,ub=max_val['SOLAR'])
        ## hydro generators
        if self.pm.opt_hydro_daily:
            # spillage
            self.var_SPILLAGE = self.gm.addVars(self.set_HYDRO_AREA,self.set_DAYS,vtype=GRB.CONTINUOUS,name='SPILLAGE',lb=0,ub=np.inf)
            # reservoir storage
            self.var_RES = self.gm.addVars(self.set_HYDRO_AREA,self.set_DAYS,vtype=GRB.CONTINUOUS,name='RES',lb=0,ub=max_val['RES'])
        else:
            # spillage
            self.var_SPILLAGE = self.gm.addVars(self.set_HYDRO_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,name='SPILLAGE',lb=0,ub=np.inf)
            # reservoir storage
            self.var_RES = self.gm.addVars(self.set_HYDRO_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,name='RES',lb=0,ub=max_val['RES'])
        # run of river hydro
        self.var_HROR = self.gm.addVars(self.set_ROR_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,name='HROR',lb=0,ub=max_val['HROR'])
        ## internal transmission
        self.var_X1 = self.gm.addVars(self.set_XINT,self.set_TIME,vtype=GRB.CONTINUOUS,name='X1',lb=0,ub=max_val['X1'])
        self.var_X2 = self.gm.addVars(self.set_XINT,self.set_TIME,vtype=GRB.CONTINUOUS,name='X2',lb=0,ub=max_val['X2'])
        # external transmission to areas with fixed price
        self.var_XEXT = self.gm.addVars(self.set_XEXT_VAR,self.set_TIME,vtype=GRB.CONTINUOUS,name='XEXT',lb=min_val['XEXT'],ub=max_val['XEXT'])
        # pumping
        self.var_PUMP = self.gm.addVars(self.set_PUMP_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,lb=0.0,ub=max_val['PUMP'])
        # if m.opt_pump_separate_reservoir:
        self.var_REL = self.gm.addVars(self.set_PUMP_RES_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,lb=0.0,ub=np.inf) # upper limit from total hydro lim
        self.var_PRES = self.gm.addVars(self.set_PUMP_RES_AREA,self.set_TIME,vtype=GRB.CONTINUOUS,lb=0.0,ub=max_val['PRES'])

        self.constr_sets = {
            'POWER_BALANCE':['AREA','TIME'],
            'HVDC_RAMP_UP':['HVDC','TIME'],
            'HVDV_RAMP_DW':['HVDC','TIME'],
            'HVDC_RAMP_EXT_UP':['XEXT_VAR','TIME'],
            'HVDC_RAMP_EXT_DW':['XEXT_VAR','TIME'],
            'GEN_RAMP_UP':['COMB_GEN','TIME'],
            'GEN_RAMP_DW':['COMB_GEN','TIME'],
            'FIX_RESERVOIR':['HYDRO_AREA'],
            'INERTIA':['TIME'],
        }
        if self.pm.opt_hydro_daily:
            self.constr_sets['RESERVOIR_BALANCE'] = ['HYDRO_AREA','DAYS']
        else:
            self.constr_sets['RESERVOIR_BALANCE'] = ['HYDRO_AREA','TIME']

        if self.pm.opt_country_reserves:
            self.constr_sets['RESERVES_UP'] = ['RESERVE_COUNTRY','TIME']
            self.constr_sets['RESERVES_DW'] = ['RESERVE_COUNTRY','TIME']
        else:
            self.constr_sets['RESERVES_UP'] = ['RESERVE_AREA','TIME']
            self.constr_sets['RESERVES_DW'] = ['RESERVE_AREA','TIME']

        if prt:
            print('Setting up CONSTRAINTS')

        # choose generator costs
        if m.opt_use_var_cost:
            if highres:
                self.gen_c1 = m.gen_c1_hr
                self.gen_c2 = m.gen_c2_hr
            elif delta is None:
                self.gen_c1 = m.gen_c1
                self.gen_c2 = m.gen_c2
            else:
                self.gen_c1 = m.gen_c1_lr
                self.gen_c2 = m.gen_c2_lr

        ## OBJECTIVE
        self.setup_objective()
        self.setup_power_balance()
        if m.opt_hydro_daily:
            self.setup_reservoir_balance_daily()
        else:
            self.setup_reservoir_balance()
        self.setup_hydro_lim()
        self.setup_pump_balance()
        self.setup_hvdc_ramp()
        self.setup_hvdc_ramp_ext()
        self.setup_thermal_ramp()
        self.setup_hydro_ramp()
        if m.opt_hydro_daily:
            self.setup_fix_reservoir_daily()
        else:
            self.setup_fix_reservoir()
        if m.opt_use_reserves:
            self.setup_reserves()
        if m.opt_use_inertia_constr:
            self.setup_inertia()

        self.gm.update()

    def setup_power_balance(self):
        m=self.pm

        self.constr_POWER_BALANCE = {}
        expr = gp.LinExpr()
        for a in self.set_AREA:
            for t in self.set_TIME:
                # add production
                expr.addTerms(
                    [1 for g in self.set_AREA_TO_GEN[a]],
                    [self.var_PG[g,t] for g in self.set_AREA_TO_GEN[a]]
                )
                # internal exports
                expr.addTerms(
                    [-1 for c in self.set_XINT_FW[a]],
                    [self.var_X1[c,t] for c in self.set_XINT_FW[a]]
                )
                expr.addTerms(
                    [-1 for c in self.set_XINT_BW[a]],
                    [self.var_X2[c,t] for c in self.set_XINT_BW[a]]
                )
                # internal imports
                expr.addTerms(
                    [(1-m.opt_loss) for c in self.set_XINT_BW[a]],
                    [self.var_X1[c,t] for c in self.set_XINT_BW[a]]
                )
                expr.addTerms(
                    [(1-m.opt_loss) for c in self.set_XINT_FW[a]],
                    [self.var_X2[c,t] for c in self.set_XINT_FW[a]]
                )
                # fixed external transfer
                expr.addConstant(
                    -MWtoGW*sum( m.exchange.at[m.timerange[t],m.xtrans_ext.at[c,'label_fw']]
                                 for c in self.set_AREA_TO_XEXT_PAR[a] )
                )
                # variable external transfer
                expr.addTerms(
                    [-1 for c in self.set_AREA_TO_XEXT_VAR[a]],
                    [self.var_XEXT[c,t] for c in self.set_AREA_TO_XEXT_VAR[a]]
                )
                # load shedding
                expr.addTerms(1,self.var_LS[a,t])
                # nominal load
                expr.addConstant(-self.max_val['LS'][a,t])
                if a in self.set_WIND_AREA:
                    expr.addTerms(1,self.var_WIND[a,t])
                if a in self.set_SOLAR_AREA:
                    expr.addTerms(1,self.var_SOLAR[a,t])
                if a in self.set_ROR_AREA:
                    expr.addTerms(1,self.var_HROR[a,t])
                if a in self.set_PUMP_AREA:
                    expr.addTerms(-1,self.var_PUMP[a,t])
                if a in self.set_PUMP_RES_AREA:
                    # separate pump production
                    expr.addTerms(1,self.var_REL[a,t])

                # add constraint
                self.constr_POWER_BALANCE[(a,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.EQUAL,
                    rhs=0,
                    name=f'POWER_BALANCE[{a},{t}]'
                )
                expr.clear()

    def setup_reservoir_balance_daily(self):
        m=self.pm
        scale = GWtoTW
        self.constr_RESERVOIR_BALANCE = {}
        expr = gp.LinExpr()
        for a in self.set_HYDRO_AREA:
            for t in self.set_DAYS:

                # initial reservoir content
                if t == 0:
                    expr.addConstant(m.reservoir_fix.at[m.daysrange_p1[0],a])
                else:
                    expr.addTerms(1,self.var_RES[a,t-1])
                # spillage
                expr.addTerms(-scale,self.var_SPILLAGE[a,t])
                # production
                expr.addTerms(
                    [-scale for g in self.set_HYDRO_AREA_TO_GEN[a] for tt in m.day2hour[t]],
                    [self.var_PG[g,tt] for g in self.set_HYDRO_AREA_TO_GEN[a] for tt in m.day2hour[t]]
                )
                # pumping in main reservoir
                if a in self.set_PUMP_NORES_AREA:
                    expr.addTerms(
                        [scale*m.opt_pump_efficiency for tt in m.day2hour[t]],
                        [self.var_PUMP[a,tt] for tt in m.day2hour[t]]
                    )
                # inflow
                if a in self.set_ROR_AREA:
                    expr.addConstant(scale*max(0,m.inflow_daily.at[m.daysrange_p1[t],a] \
                                         - MWtoGW*m.ror_daily.at[m.daysrange_p1[t],a]))
                else:
                    # use all inflow
                    expr.addConstant(scale*m.inflow_daily.at[m.daysrange_p1[t],a])

                self.constr_RESERVOIR_BALANCE[(a,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.EQUAL,
                    rhs=self.var_RES[a,t],
                    name=f'RESERVOIR_BALANCE[{a},{t}]'
                )
                expr.clear()

    def setup_reservoir_balance(self):
        pass
        m=self.pm
        self.constr_RESERVOIR_BALANCE = {}
        expr = gp.LinExpr()
        scale = GWtoTW
        for a in self.set_HYDRO_AREA:
            for t in self.set_TIME:

                # initial reservoir content
                if t == self.set_TIME[0]:
                    expr.addConstant(m.reservoir_fix.at[m.timerange[t],a])
                else:
                    expr.addTerms(1,self.var_RES[a,t-1])
                # spillage
                expr.addTerms(-1*scale,self.var_SPILLAGE[a,t])
                # production
                expr.addTerms(
                    [-scale for g in self.set_HYDRO_AREA_TO_GEN[a]],
                    [self.var_PG[g,t] for g in self.set_HYDRO_AREA_TO_GEN[a]]
                )
                # pumping in main reservoir
                if a in self.set_PUMP_NORES_AREA:
                    expr.addTerms(scale*self.pm.opt_pump_efficiency,self.var_PUMP[a,t])
                # inflow
                if a in self.set_ROR_AREA:
                    # val = max(0,
                    #           m.inflow_hourly.at[m.timerange[t],a] - MWtoGW*m.ror_hourly.at[m.timerange[t],a] )
                    # if a in ['SE1','SE2'] and t == 0:
                    #     print(f'Adding inflow for {a} and t={t}: {val:0.4f}')
                    expr.addConstant(scale*max(0,
                                         m.inflow_hourly.at[m.timerange[t],a] - MWtoGW*m.ror_hourly.at[m.timerange[t],a] )
                                     )
                else:
                    # use all inflow
                    expr.addConstant(scale*m.inflow_hourly.at[m.timerange[t],a])
                self.constr_RESERVOIR_BALANCE[(a,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.EQUAL,
                    rhs=self.var_RES[a,t],
                    name=f'RESERVOIR_BALANCE[{a},{t}]'
                )
                expr.clear()

    def setup_pump_balance(self):
        """ Reservoir balance for pumping, separate reservoirs """
        pass
        m=self.pm
        scale = GWtoTW

        self.constr_PUMP_BALANCE = {}
        expr = gp.LinExpr()
        for a in self.set_PUMP_RES_AREA:
            for t in self.set_TIME:
                # initial reservoir content
                if t == self.set_TIME[0]:
                    expr.addConstant(0)
                else:
                    expr.addTerms(1,self.var_PRES[a,t-1])

                # pumping and discharge (release)
                expr.addTerms([-scale,scale*m.opt_pump_efficiency],[self.var_REL[a,t],self.var_PUMP[a,t]])

                self.constr_PUMP_BALANCE[(a,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.EQUAL,
                    rhs=self.var_PRES[a,t],
                    name=f'PUMP_BALANCE[{a},{t}]'
                )
                expr.clear()

    def setup_hydro_lim(self):
        m = self.pm
        expr = gp.LinExpr()
        self.constr_MAX_HYDRO = {}
        self.constr_MIN_HYDRO = {}
        for a in self.set_HYDRO_AREA:
            hgens = [g for g in self.set_HYDRO_GEN if m.gen_data.at[g,'area'] == a]
            # hgens = m.gen_data.loc[(m.gen_data['gtype']=='Hydro')&(m.gen_data['area']==a),['pmax','pmin']]
            hmax = m.gen_data.loc[hgens,'pmax'].sum() * MWtoGW
            hmin = m.gen_data.loc[hgens,'pmin'].sum() * MWtoGW
            for t in self.set_TIME:
                expr.addTerms([1 for g in hgens],[self.var_PG[g,t] for g in hgens])
                if a in self.set_ROR_AREA:
                    expr.addTerms(1,self.var_HROR[a,t])
                if a in self.set_PUMP_RES_AREA:
                    expr.addTerms(1,self.var_REL[a,t])
                self.constr_MIN_HYDRO[(a,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.GREATER_EQUAL,
                    rhs=hmin,
                    name=f'MIN_HYDRO[{a,t}]'
                )
                self.constr_MAX_HYDRO[(a,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.LESS_EQUAL,
                    rhs=hmax,
                    name=f'MAX_HYDRO[{a,t}]'
                )
                expr.clear()

    def setup_hvdc_ramp_ext(self):
        m = self.pm
        expr = gp.LinExpr()
        self.constr_HVDC_RAMP_EXT_UP = {}
        self.constr_HVDC_RAMP_EXT_DW = {}

        for c in self.set_XEXT_VAR:
            for t in self.set_TIME[1:]:

                expr.addTerms([1,-1],[self.var_XEXT[c,t],self.var_XEXT[c,t-1]])
                self.constr_HVDC_RAMP_EXT_UP[(c,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.LESS_EQUAL,
                    rhs = m.xtrans_ext.at[c,'ramp']*MWtoGW,
                    name = f'HVDC_RAMP_EXT_UP[{c},{t}]'
                )
                self.constr_HVDC_RAMP_EXT_DW[(c,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.GREATER_EQUAL,
                    rhs = -m.xtrans_ext.at[c,'ramp']*MWtoGW,
                    name = f'HVDC_RAMP_EXT_DW[{c},{t}]'
                )
                expr.clear()

    def setup_hvdc_ramp(self):
        m = self.pm
        self.constr_HVDC_RAMP_UP = {}
        self.constr_HVDC_RAMP_DW = {}
        expr = gp.LinExpr()
        for c in self.set_HVDC:
            for t in self.set_TIME[1:]:
                expr.addTerms(
                    [1 for conn in self.set_HVDC_TO_XINT[c]],
                    [self.var_X1[conn,t] for conn in self.set_HVDC_TO_XINT[c]]
                )
                expr.addTerms(
                    [-1 for conn in self.set_HVDC_TO_XINT[c]],
                    [self.var_X2[conn,t] for conn in self.set_HVDC_TO_XINT[c]]
                )
                expr.addTerms(
                    [-1 for conn in self.set_HVDC_TO_XINT[c]],
                    [self.var_X1[conn,t-1] for conn in self.set_HVDC_TO_XINT[c]]
                )
                expr.addTerms(
                    [1 for conn in self.set_HVDC_TO_XINT[c]],
                    [self.var_X2[conn,t-1] for conn in self.set_HVDC_TO_XINT[c]]
                )
                # expr.addTerms([1,-1,-1,1],[self.var_X1[c,t],self.var_X2[c,t],self.var_X1[c,t-1],self.var_X2[c,t-1]])
                self.constr_HVDC_RAMP_UP[(c,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.LESS_EQUAL,
                    rhs=m.opt_hvdc_max_ramp*MWtoGW,
                    name=f'HVDC_RAMP_UP[{c},{t}]'
                )
                self.constr_HVDC_RAMP_DW[(c,t)] = self.gm.addLConstr(
                    lhs=expr,
                    sense=GRB.GREATER_EQUAL,
                    rhs=-m.opt_hvdc_max_ramp*MWtoGW,
                    name=f'HVDC_RAMP_DW[{c},{t}]'
                )
                expr.clear()

    def setup_hydro_ramp(self):
        pass
        m = self.pm
        self.constr_HYDRO_RAMP_UP = {}
        self.constr_HYDRO_RAMP_DW = {}
        expr = gp.LinExpr()
        for a in self.set_HYDRO_AREA:
            hgens = [g for g in self.set_GEN if m.gen_data.at[g,'area'] == a and \
                     m.gen_data.at[g,'gtype'] == 'Hydro']
            max_ramp = m.gen_data.loc[hgens,'rampup'].sum()
            min_ramp = m.gen_data.loc[hgens,'rampdown'].sum()
            for t in self.set_TIME[1:]:
                expr.addTerms( # reservoir hydro
                    [1 for g in hgens],
                    [self.var_PG[g,t] for g in hgens]
                )
                expr.addTerms(
                    [-1 for g in hgens],
                    [self.var_PG[g,t-1] for g in hgens]
                )
                if a in self.set_PUMP_RES_AREA:
                    expr.addTerms([1,-1],[self.var_REL[a,t],self.var_REL[a,t-1]])
                if a in self.set_ROR_AREA:
                    expr.addTerms([1,-1],[self.var_HROR[a,t],self.var_HROR[a,t-1]])

                self.constr_HYDRO_RAMP_UP[(a,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.LESS_EQUAL,
                    rhs = max_ramp*MWtoGW,
                    name = f'HYDRO_RAMP_UP[{a},{t}]'
                )
                self.constr_HYDRO_RAMP_DW[(a,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.GREATER_EQUAL,
                    rhs = min_ramp*MWtoGW,
                    name = f'HYDRO_RAMP_DW[{a},{t}]'
                )
                expr.clear()

    def setup_thermal_ramp(self):
        m = self.pm
        self.constr_GEN_RAMP_UP = {}
        self.constr_GEN_RAMP_DW = {}
        expr = gp.LinExpr()
        for g in self.set_THERMAL_GEN: # includes nuclear
            for t in self.set_TIME[1:]:
                expr.addTerms(
                    [1,-1],
                    [self.var_PG[g,t], self.var_PG[g,t-1]]
                )
                self.constr_GEN_RAMP_UP[(g,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.LESS_EQUAL,
                    rhs = m.gen_data.at[g,'rampup']*MWtoGW,
                    name = f'GEN_RAMP_UP[{g},{t}]'
                )
                self.constr_GEN_RAMP_DW[(g,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.GREATER_EQUAL,
                    rhs = m.gen_data.at[g,'rampdown']*MWtoGW,
                    name = f'GEN_RAMP_DW[{g},{t}]'
                )
                expr.clear()

    def setup_gen_ramp(self):
        m = self.pm
        self.constr_GEN_RAMP_UP = {}
        self.constr_GEN_RAMP_DW = {}
        expr = gp.LinExpr()
        for gc in self.set_COMB_GEN:
            for t in self.set_TIME[1:]:
                expr.addTerms(
                    [1 for g in self.set_COMB_GEN_TO_GEN[gc]],
                    [self.var_PG[g,t] for g in self.set_COMB_GEN_TO_GEN[gc]]
                )
                expr.addTerms(
                    [-1 for g in self.set_COMB_GEN_TO_GEN[gc]],
                    [self.var_PG[g,t-1] for g in self.set_COMB_GEN_TO_GEN[gc]]
                )
                self.constr_GEN_RAMP_UP[(gc,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.LESS_EQUAL,
                    rhs = m.gen_data_comb.at[gc,'rampup']*MWtoGW,
                    name = f'GEN_RAMP_UP[{gc},{t}]'
                )
                self.constr_GEN_RAMP_DW[(gc,t)] = self.gm.addLConstr(
                    lhs = expr,
                    sense = GRB.GREATER_EQUAL,
                    rhs = m.gen_data_comb.at[gc,'rampdown']*MWtoGW,
                    name = f'GEN_RAMP_DW[{gc},{t}]'
                )
                expr.clear()

    def setup_fix_reservoir(self):
        m = self.pm
        self.constr_FIX_RESERVOIR = {}
        for a in self.set_HYDRO_AREA:
            self.constr_FIX_RESERVOIR[a] = self.gm.addLConstr(
                lhs=self.var_RES[a,self.set_TIME[-1]],
                sense=GRB.EQUAL,
                rhs=m.reservoir_fix.at[str_to_date(m.endtime),a],
                name=f'FIX_RESERVOIR[{a}]'
            )

    def setup_fix_reservoir_daily(self):
        m = self.pm
        self.constr_FIX_RESERVOIR = {}
        for a in self.set_HYDRO_AREA:
            self.constr_FIX_RESERVOIR[a] = self.gm.addLConstr(
                lhs=self.var_RES[a,self.set_DAYS[-1]],
                sense=GRB.EQUAL,
                rhs=m.reservoir_fix.at[m.daysrange_p1[-1],a],
                name=f'FIX_RESERVOIR[{a}]'
            )

    def setup_reserves(self):
        # Note: currently only includes hydro generators
        m = self.pm
        self.constr_RESERVES_UP = {}
        self.constr_RESERVES_DW = {}
        expr = gp.LinExpr()
        if m.opt_country_reserves:
            for c in self.set_RESERVE_COUNTRY:
                for t in self.set_TIME:
                    # upward reserves
                    expr.addTerms(
                        [-1 for g in self.set_RESERVE_COUNTRY_TO_GEN[c]],
                        [self.var_PG[g,t] for g in self.set_RESERVE_COUNTRY_TO_GEN[c]]
                    )
                    expr.addConstant(MWtoGW*sum(m.gen_data.at[g,'pmax'] for g in self.set_RESERVE_COUNTRY_TO_GEN[c]))
                    if c in self.set_ROR_COUNTRY:
                        expr.addTerms(
                            [-1 for a in m.country_to_areas[c]],
                            [self.var_HROR[a,t] for a in m.country_to_areas[c]]
                        )
                    self.constr_RESERVES_UP[(c,t)] = self.gm.addLConstr(
                        lhs = expr,
                        sense = GRB.GREATER_EQUAL,
                        rhs = m.reserve_country_data.at[c,'Rp']*MWtoGW,
                        name = f'RESERVES_UP[{c},{t}]'
                    )
                    expr.clear()

                    # downward reserves
                    expr.addTerms(
                        [1 for g in self.set_RESERVE_COUNTRY_TO_GEN[c]],
                        [self.var_PG[g,t] for g in self.set_RESERVE_COUNTRY_TO_GEN[c]]
                    )
                    expr.addConstant(-MWtoGW*sum(m.gen_data.at[g,'pmin'] for g in self.set_RESERVE_COUNTRY_TO_GEN[c]))
                    if c in self.set_ROR_COUNTRY:
                        expr.addTerms(
                            [1 for a in m.country_to_areas[c]],
                            [self.var_HROR[a,t] for a in m.country_to_areas[c]]
                        )
                    self.constr_RESERVES_DW[(c,t)] = self.gm.addLConstr(
                        lhs = expr,
                        sense = GRB.GREATER_EQUAL,
                        rhs = m.reserve_country_data.at[c,'Rn']*MWtoGW,
                        name = f'RESERVES_DW[{c},{t}]'
                    )
                    expr.clear()
        else:
            for a in self.set_RESERVE_AREA:
                for t in self.set_TIME:
                    pass
                    # upward reserves
                    expr.addTerms(
                        [-1 for g in self.set_RESERVE_AREA_TO_GEN[a]],
                        [self.var_PG[g,t] for g in self.set_RESERVE_AREA_TO_GEN[a]]
                    )
                    expr.addConstant(MWtoGW*sum(m.gen_data.at[g,'pmax'] for g in self.set_RESERVE_AREA_TO_GEN[a]))
                    if a in self.set_ROR_AREA:
                        expr.addTerms(-1,self.var_HROR[a,t])

                    self.constr_RESERVES_UP[(a,t)] = self.gm.addLConstr(
                        lhs = expr,
                        sense = GRB.GREATER_EQUAL,
                        rhs = m.reserve_data.at[a,'Rp']*MWtoGW,
                        name = f'RESERVES_UP[{a},{t}]'
                    )
                    expr.clear()

                    # downward reserves
                    expr.addTerms(
                        [1 for g in self.set_RESERVE_AREA_TO_GEN[a]],
                        [self.var_PG[g,t] for g in self.set_RESERVE_AREA_TO_GEN[a]]
                    )
                    expr.addConstant(-MWtoGW*sum(m.gen_data.at[g,'pmin'] for g in self.set_RESERVE_AREA_TO_GEN[a]))
                    if a in self.set_ROR_AREA:
                        expr.addTerms(1,self.var_HROR[a,t])

                    self.constr_RESERVES_DW[(a,t)] = self.gm.addLConstr(
                        lhs = expr,
                        sense = GRB.GREATER_EQUAL,
                        rhs = m.reserve_data.at[a,'Rn']*MWtoGW,
                        name = f'RESERVES_DW[{a},{t}]'
                    )
                    expr.clear()

    def setup_inertia(self):
        m = self.pm
        pass

        self.constr_INERTIA = {}
        expr = gp.LinExpr()
        for t in self.set_TIME:
            for g in self.set_GEN:
                a = m.gen_data.at[g,'area']
                gtype = m.gen_data.at[g,'gtype']
                c = m.area_to_country[a]
                if a in self.set_SYNCAREA:
                    expr.addTerms(
                        m.opt_inertia_constants[c][gtype] / m.opt_inertia_cf[c][gtype] / m.opt_inertia_pf[c][gtype],self.var_PG[g,t]
                    )
            for a in self.set_ROR_AREA:
                if a in self.set_SYNCAREA:
                    gtype = 'Hydro'
                    c = m.area_to_country[a]
                    expr.addTerms(
                        m.opt_inertia_constants[c][gtype] / m.opt_inertia_cf[c][gtype] / m.opt_inertia_pf[c][gtype],self.var_HROR[a,t]
                    )
            for a in self.set_PUMP_RES_AREA:
                if a in self.set_SYNCAREA:
                    gtype = 'Hydro'
                    c = m.area_to_country[a]
                    expr.addTerms(
                        m.opt_inertia_constants[c][gtype] / m.opt_inertia_cf[c][gtype] / m.opt_inertia_pf[c][gtype],self.var_REL[a,t]
                    )
            self.constr_INERTIA[t] = self.gm.addLConstr(
                lhs = expr,
                sense = GRB.GREATER_EQUAL,
                rhs = m.opt_min_kinetic_energy,
                name = f'INERTIA[{t}]',
            )
            expr.clear()

    def setup_objective(self):
        m = self.pm
        obj = gp.QuadExpr()

        # generator costs
        if m.opt_use_var_cost:
            obj.addTerms(
                [self.gen_c2.at[m.timerange[t],g] for g in self.set_THERMAL_GEN for t in self.set_TIME],
                [self.var_PG[g,t] for g in self.set_THERMAL_GEN for t in self.set_TIME],
                [self.var_PG[g,t] for g in self.set_THERMAL_GEN for t in self.set_TIME]
            )
            obj.addTerms(
                [self.gen_c1.at[m.timerange[t],g]*MWtoGW for g in self.set_THERMAL_GEN for t in self.set_TIME],
                [self.var_PG[g,t] for g in self.set_THERMAL_GEN for t in self.set_TIME]
            )
            if m.opt_hydro_cost:
                obj.addTerms(
                    [self.gen_c2.at[m.timerange[t],g] for g in self.set_HYDRO_GEN for t in self.set_TIME],
                    [self.var_PG[g,t] for g in self.set_HYDRO_GEN for t in self.set_TIME],
                    [self.var_PG[g,t] for g in self.set_HYDRO_GEN for t in self.set_TIME]
                )
                obj.addTerms(
                    [self.gen_c1.at[m.timerange[t],g]*MWtoGW for g in self.set_HYDRO_GEN for t in self.set_TIME],
                    [self.var_PG[g,t] for g in self.set_HYDRO_GEN for t in self.set_TIME]
                )
        else:
            obj.addTerms(
                [m.gen_data.at[g,'c2'] for g in self.set_THERMAL_GEN for t in self.set_TIME],
                [self.var_PG[g,t] for g in self.set_THERMAL_GEN for t in self.set_TIME],
                [self.var_PG[g,t] for g in self.set_THERMAL_GEN for t in self.set_TIME]
            )
            obj.addTerms(
                [m.gen_data.at[g,'c1']*MWtoGW for g in self.set_THERMAL_GEN for t in self.set_TIME],
                [self.var_PG[g,t] for g in self.set_THERMAL_GEN for t in self.set_TIME]
            )

        obj.addTerms(
            [m.opt_loadshed_cost*MWtoGW for a in self.set_AREA for t in self.set_TIME],
            [self.var_LS[a,t] for a in self.set_AREA for t in self.set_TIME]
        )

        # net imports
        obj.addTerms(
            [-MWtoGW*m.price_external.at[m.timerange[t],m.xtrans_ext.at[x,'to']] for x in self.set_XEXT_VAR for t in self.set_TIME],
            [self.var_XEXT[x,t] for x in self.set_XEXT_VAR for t in self.set_TIME]
        )
        # cost for wind power
        obj.addTerms(
            [MWtoGW*m.opt_wind_cost for a in self.set_WIND_AREA for t in self.set_TIME],
            [self.var_WIND[a,t] for a in self.set_WIND_AREA for t in self.set_TIME]
        )

        self.gm.setObjective(obj,sense=GRB.MINIMIZE)

    def get_variable(self,name):
        """ Construct pandas dataframe from parameter or variable
          Call after solving model to get data into dataframes

          Note: name is of form "var_XXX"
        """
        name = name.split('_',1)[1]

        sets = self.var_sets[name]
        entity = self.__getattribute__(f'var_{name}')
        cols = self.__getattribute__(f'set_{sets[0]}')

        if sets.__len__() == 2:
            index = self.__getattribute__(f'set_{sets[1]}')
            df = pd.DataFrame(dtype=float,index=index,columns=cols)
            # Note: using getAttr to create a dict with values is faster than iterating over each element and using .X
            res = self.gm.getAttr('x',entity)
            for t in index:
                for c in cols:
                    df.at[t,c] = res[(c,t)]
        else:
            print(f'Failed to fetch {name}, variables of {sets.__len__()} sets not implemented!')
            df = None

        return df

    def get_dual(self,name):
        """
        Get dual variable

        :param name: of form "constr_XXX"
        :return:
        """
        name = name.split('_',1)[1]

        sets = self.constr_sets[name]
        entity = self.__getattribute__(f'constr_{name}')
        cols = self.__getattribute__(f'set_{sets[0]}')

        #%%
        if sets.__len__() == 1:
            set = sets[0]
            if set == 'TIME':
                df = pd.Series(dtype=float,index=self.set_TIME)
            else:
                df = pd.Series(dtype=float,index=cols)
            res = self.gm.getAttr('Pi',entity)
            if set == 'TIME':
                for t in self.set_TIME:
                    df.at[t] = res[t]
            else:
                for c in cols:
                    df.at[c] = res[c]
        elif sets.__len__() == 2:
            index = self.__getattribute__(f'set_{sets[1]}')
            df = pd.DataFrame(dtype=float,index=index,columns=cols)
            # Note: using getAttr to create a dict with values is faster than iterating over each element
            res = self.gm.getAttr('Pi',entity)
            for t in index:
                for c in cols:
                    df.at[t,c] = res[(c,t)]
        else:
            print(f'Failed to fetch {name}, duals of {sets.__len__()} sets not implemented!')
            df = None

        return df

    def get_objective_value(self):
        return self.gm.getObjective().getValue()


    def update_inflow(self):
        """ Update values for inflow parameter """
        pass
        # remove old constraint
        self.gm.remove(self.constr_RESERVOIR_BALANCE)
        # create new constraint
        if self.pm.opt_hydro_daily:
            self.setup_reservoir_balance_daily()
        else:
            self.setup_reservoir_balance()

    def update_ror(self,dic):
        for a,t in dic:
            self.var_HROR[a,t].ub = dic[(a,t)]

    def update_wind(self,dic):
        for a,t in dic:
            self.var_WIND[a,t].ub = dic[(a,t)]

    def update_solar(self,dic):
        for a,t in dic:
            self.var_SOLAR[a,t].ub = dic[(a,t)]
