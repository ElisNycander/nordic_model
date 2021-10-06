

import pandas as pd
import pyomo.environ as pye
from pyomo.core.base.PyomoModel import ConcreteModel
from model_definitions import MWtoGW

## RULES FOR PYOMO ###
from pyomo_model_rules import plim_LS, pini_LS, plim_HROR, plim_RES, plim_PG, pini_PG, \
    pini_SOLAR, plim_SOLAR, pini_W, plim_W, pini_RES, pconstr_POWER_BALANCE, \
    plim_X1, plim_X2, plim_XEXT, pini_HROR, pini_X1, pini_X2, pini_XEXT, plim_PUMP, pini_PUMP, \
    plim_PRES, plim_REL, pconstr_PUMP_BALANCE, \
    pparam_INFLOW, \
    pobj_THERMAL_COST, pconstr_MIN_HYDRO, pconstr_MAX_HYDRO, \
    pconstr_RESERVOIR_BALANCE, pconstr_HVDC_RAMP, pconstr_HVDC_RAMP_EXT, \
    pconstr_FINAL_RESERVOIR, pconstr_RESERVES_UP, pconstr_RESERVES_DW, pconstr_PG_RAMP, pconstr_INERTIA, \
    pconstr_RESERVOIR_BALANCE_DAILY, pconstr_FINAL_RESERVOIR_DAILY, pconstr_HYDRO_RAMP, pconstr_THERMAL_RAMP

class PyomoModel(ConcreteModel):
    """ Wrapper class pyomo
    All parts of the modelling related to the api of the pyomo model goes here
    """
    def __init__(self):
        # call ConcreteModel constructor
        super(PyomoModel,self).__init__()

    def setup_opt_problem(self,m):
        """
        Setup optimization, assumes all necessary data structures have been created
        Note: m is the owner of the PyomoModel, which contains all options and
        data-structures needed to setup the optimization problem

        # naming conventions:
        set_XXX - set
        par_XXX - parameter
        var_XXX - variable
        """
        # Set parent model "pm"
        self.pm = m
        prt = m.opt_print['setup']

        ## SETS
        self.set_TIME = pye.Set(initialize=m.idx_time)
        self.set_AREA = pye.Set(initialize=m.areas)
        self.set_COUNTRY = pye.Set(initialize=m.opt_countries)
        self.set_C2A = pye.Set(self.set_COUNTRY,within=self.set_AREA,initialize=m.country_to_areas)
        self.set_SYNCAREA = pye.Set(initialize=m.syncareas)
        self.set_ROR_AREA = pye.Set(initialize=m.ror_areas)
        self.set_ROR_COUNTRY = pye.Set(initialize=m.ror_countries)
        self.set_PUMP_AREA = pye.Set(initialize=m.pump_areas)
        self.set_PUMP_RES_AREA = pye.Set(initialize=m.pump_res_areas)
        self.set_PUMP_NORES_AREA = pye.Set(initialize=m.pump_nores_areas)

        self.set_DAYS = pye.Set(initialize=m.idx_day)

        # all gens
        self.set_GEN = pye.Set(initialize = [i for i in range(1,m.nGen+1)])
        # hydro gens
        self.set_HYDRO_GEN = pye.Set(within = self.set_GEN,initialize = [i for i in range(1,m.nGen+1) if m.gen_data.at[i,'gtype'] == 'Hydro'])
        # all gens - hydro gens
        self.set_THERMAL_GEN = pye.Set(within = self.set_GEN,initialize = [i for i in range(1,m.nGen+1) if not m.gen_data.at[i,'gtype'] == 'Hydro'])
        # nuclear
        self.set_NUCLEAR_GEN = pye.Set(within=self.set_GEN,initialize=[i for i in range(1,m.nGen+1) if m.gen_data.at[i,'gtype'] == 'Nuclear'])
        # define set of combined generators
        self.set_COMB_GEN = pye.Set(initialize = [i for i in range(1,m.nGenComb+1)])
        self.set_COMB_GEN_TO_GEN = pye.Set(self.set_COMB_GEN,within=self.set_GEN,initialize = m.gen_comb)
        self.set_RESERVE_AREA = pye.Set(initialize = m.resareas)
        self.set_RESERVE_AREA_TO_GEN = pye.Set(self.set_RESERVE_AREA,within=self.set_GEN,initialize = m.reserve_gens)
        self.set_RESERVE_COUNTRY = pye.Set(initialize = [c for c in m.opt_reserves_fcrn if m.opt_reserves_fcrn[c] > 0])
        self.set_RESERVE_COUNTRY_TO_GEN = pye.Set(self.set_RESERVE_COUNTRY,within=self.set_GEN,initialize=m.reserve_gens_country)

        self.set_WIND_AREA = pye.Set(domain=self.set_AREA,initialize =m.wind_areas)
        self.set_AREA_TO_GEN = pye.Set(self.set_AREA,within=self.set_GEN,initialize=m.gen_in_area)

        self.set_HYDRO_AREA = pye.Set(within = self.set_AREA,initialize = m.hydrores)
        self.set_HYDRO_AREA_TO_GEN = pye.Set(self.set_HYDRO_AREA,within=self.set_HYDRO_GEN,initialize=m.reservoir2hydro)
        # all areas with positive solar capacity
        self.set_SOLAR_AREA = pye.Set(domain=self.set_AREA,initialize=m.solar_areas)

        # internal connections
        self.set_XINT = pye.Set(initialize = [i for i in range(1,m.nXint+1)])
        self.set_XINT_FW = pye.Set(self.set_AREA,within=self.set_XINT,initialize=m.xintf)
        self.set_XINT_BW = pye.Set(self.set_AREA,within=self.set_XINT,initialize=m.xintr)

        # external connections
        self.set_XEXT = pye.Set(initialize = [i for i in range(1,m.nXext+1)])
        # divide external connections into fixed price and fixed transfer
        # fixed price connections
        self.set_XEXT_VAR = pye.Set(initialize = m.fixed_price_connections)
        # fixed transfer connections
        self.set_XEXT_PAR = pye.Set(initialize = m.fixed_transfer_connections)
        # connections to modeled external regions
        self.set_AREA_TO_XEXT_PAR = pye.Set(self.set_AREA,within=self.set_XEXT,initialize=m.xext_ft)
        self.set_AREA_TO_XEXT_VAR = pye.Set(self.set_AREA,within=self.set_XEXT,initialize=m.xext_fp)

        self.set_HVDC = pye.Set(initialize = [i for i in m.combined_hvdc.keys()])
        self.set_HVDC_TO_XINT = pye.Set(self.set_HVDC,within=self.set_XINT,initialize = m.combined_hvdc)

        ## PARAMETERS
        if m.opt_hydro_daily:
            self.param_INFLOW = pye.Param(self.set_HYDRO_AREA,self.set_DAYS,initialize=pparam_INFLOW,mutable=True)
        else:
            self.param_INFLOW = pye.Param(self.set_HYDRO_AREA,self.set_TIME,initialize=pparam_INFLOW,mutable=True)

        ## VARIABLES
        def tag_var():
            pass
        if prt:
            print('Setting up VARIABLES')

        # load shedding
        self.var_LS = pye.Var(self.set_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_LS,initialize=pini_LS)
        ## all generators
        # production
        self.var_PG = pye.Var(self.set_GEN,self.set_TIME,domain=pye.Reals,bounds=plim_PG,initialize=pini_PG)
        # wind generation
        self.var_WIND = pye.Var(self.set_WIND_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_W,initialize=pini_W)
        # solar generation
        self.var_SOLAR = pye.Var(self.set_SOLAR_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_SOLAR,initialize=pini_SOLAR)
        ## hydro generators
        if m.opt_hydro_daily:
            # spillage
            self.var_SPILLAGE = pye.Var(self.set_HYDRO_AREA,self.set_DAYS,domain=pye.Reals,bounds=(0,None),initialize=0)
            # reservoir storage
            self.var_RES = pye.Var(self.set_HYDRO_AREA,self.set_DAYS,domain=pye.Reals,bounds=plim_RES,initialize=pini_RES)
        else:
            # spillage
            self.var_SPILLAGE = pye.Var(self.set_HYDRO_AREA,self.set_TIME,domain=pye.Reals,bounds=(0,None),initialize=0)
            # reservoir storage
            self.var_RES = pye.Var(self.set_HYDRO_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_RES,initialize=pini_RES)
        # run of river hydro
        self.var_HROR = pye.Var(self.set_ROR_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_HROR,initialize=pini_HROR)
        # pump hydro
        self.var_PUMP = pye.Var(self.set_PUMP_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_PUMP,initialize=pini_PUMP)
        self.var_REL = pye.Var(self.set_PUMP_RES_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_REL,initialize=0)
        self.var_PRES = pye.Var(self.set_PUMP_RES_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_PRES,initialize=0)

        ## internal transmission
        self.var_X1 = pye.Var(self.set_XINT,self.set_TIME,domain=pye.Reals,bounds=plim_X1,initialize=pini_X1)
        self.var_X2 = pye.Var(self.set_XINT,self.set_TIME,domain=pye.Reals,bounds=plim_X2,initialize=pini_X2)
        # external transmission to areas with fixed price
        self.var_XEXT = pye.Var(self.set_XEXT_VAR,self.set_TIME,domain=pye.Reals,bounds=plim_XEXT,initialize=pini_XEXT)

        ## OBJECTIVE

        self.OBJ = pye.Objective(rule=pobj_THERMAL_COST)

        ## CONSTRAINTS
        if prt:
            print('Setting up CONSTRAINTS')
        self.constr_POWER_BALANCE = pye.Constraint(self.set_AREA,self.set_TIME,rule=pconstr_POWER_BALANCE)

        if m.opt_hydro_daily:
            self.constr_RESERVOIR_BALANCE = pye.Constraint(self.set_HYDRO_AREA,self.set_DAYS,rule=pconstr_RESERVOIR_BALANCE_DAILY)
        else:
            self.constr_RESERVOIR_BALANCE = pye.Constraint(self.set_HYDRO_AREA,self.set_TIME,rule=pconstr_RESERVOIR_BALANCE)
        if m.opt_hydro_daily:
            self.constr_FIX_RESERVOIR = pye.Constraint(self.set_HYDRO_AREA,rule=pconstr_FINAL_RESERVOIR_DAILY)
        else:
            self.constr_FIX_RESERVOIR = pye.Constraint(self.set_HYDRO_AREA,rule=pconstr_FINAL_RESERVOIR)

        self.constr_MIN_HYDRO = pye.Constraint(self.set_HYDRO_AREA,self.set_TIME,rule=pconstr_MIN_HYDRO)
        self.constr_MAX_HYDRO = pye.Constraint(self.set_HYDRO_AREA,self.set_TIME,rule=pconstr_MAX_HYDRO)

        self.constr_PUMP_BALANCE = pye.Constraint(self.set_PUMP_RES_AREA,self.set_TIME,rule=pconstr_PUMP_BALANCE)

        self.constr_HVDC_RAMP = pye.Constraint(self.set_HVDC,self.set_TIME,rule=pconstr_HVDC_RAMP)
        # #
        self.constr_THERMAL_RAMP = pye.Constraint(self.set_THERMAL_GEN,self.set_TIME,rule=pconstr_THERMAL_RAMP)
        self.constr_HYDRO_RAMP = pye.Constraint(self.set_HYDRO_AREA,self.set_TIME,rule=pconstr_HYDRO_RAMP)
        # #
        self.constr_HVDC_RAMP_EXT = pye.Constraint(self.set_XEXT_VAR,self.set_TIME,rule=pconstr_HVDC_RAMP_EXT)

        if m.opt_use_reserves:
            if m.opt_country_reserves: # reserves by country
                self.constr_RESERVES_UP = pye.Constraint(self.set_RESERVE_COUNTRY,self.set_TIME,rule=pconstr_RESERVES_UP)
                self.constr_RESERVES_DW = pye.Constraint(self.set_RESERVE_COUNTRY,self.set_TIME,rule=pconstr_RESERVES_DW)
            else: # reserves by area
                self.constr_RESERVES_UP = pye.Constraint(self.set_RESERVE_AREA,self.set_TIME,rule=pconstr_RESERVES_UP)
                self.constr_RESERVES_DW = pye.Constraint(self.set_RESERVE_AREA,self.set_TIME,rule=pconstr_RESERVES_DW)

        if m.opt_use_inertia_constr:
            self.constr_INERTIA = pye.Constraint(self.set_TIME,rule=pconstr_INERTIA)

    def get_variable(self,name):
        """ Construct pandas dataframe from parameter or variable
        Call after solving model to get data into dataframes
        !! Should be rewritten using iteritems !!
        """
        entity = self.__getattribute__(name)
        setnames = [set.name for set in entity._index.subsets()]

        #print(setnames)

        if setnames.__len__() == 1:
            # not implemented
            index = list(set(self.__getattribute__(setnames[0])))
            res = pd.DataFrame(dtype=float,index=index)
            if isinstance(entity,pye.Param):
                for idx in index:
                    res.at[idx] = entity[idx]
            elif isinstance(entity,pye.Var):
                for idx in index:
                    res.at[idx] = entity[idx].value
            else:
                print('1D objects of type ''{0}'' not implemented'.format(entity.type))
                pass
            pass
        elif setnames.__len__() == 2:
            tindex = list(set(self.__getattribute__(setnames[1])))
            cols = self.__getattribute__(setnames[0])
            res = pd.DataFrame(index=tindex,columns=cols)

            if isinstance(entity,pye.Param):
                for col in cols:
                    for tidx in tindex:
                        res.at[tidx,col] = entity[col,tidx]
            elif isinstance(entity,pye.Var):
                for col in cols:
                    for tidx in tindex:
                        res.at[tidx,col] = entity[col,tidx].value
            else:
                print('2D objects of type ''{0}'' not implemented'.format(entity.type))
                pass
        else:
            # not implemented
            print('3D objects not implemented')
            pass

        return res

    def get_dual(self,name):
        """ Get dual variables of given constraint
                !! Should be rewritten using iteritems !!
        """
        entity = self.__getattribute__(name)
        setnames = [set.name for set in entity._index.subsets()]

        if setnames.__len__() == 1:

            index = list(set(self.__getattribute__(setnames[0])))
            res = pd.Series(dtype=float,index=index)
            for idx in index:
                res.at[idx] = self.dual[entity[idx]]
            pass
        elif setnames.__len__() == 2:
            tindex = list(set(self.__getattribute__(setnames[1])))
            cols = list(set(self.__getattribute__(setnames[0])))
            res = pd.DataFrame(index=tindex,columns=cols)
            for constr in entity.iteritems(): # better use iteritems to only loop over acitve constraints
                tidx = constr[0][1]
                cidx = constr[0][0]
                res.at[tidx,cidx] = self.dual[constr[1]]
        else:
            # not implemented
            print('3D objects not implemented')
            pass
        return res

    def get_bound(self,name,limit):
        """ Get bound of given variable
        limit: 'upper' or 'lower'
        """

        entity = self.__getattribute__(name)
        setnames = [set.name for set in entity._index.subsets()]
        if setnames.__len__() == 2:
            tindex = list(set(self.__getattribute__(setnames[1])))
            cols = list(set(self.__getattribute__(setnames[0])))
            res = pd.DataFrame(index=tindex,columns=cols)
            for val in entity.iteritems(): # better use iteritems to only loop over acitve constraints
                tidx = val[0][1]
                cidx = val[0][0]
                if limit == 'upper':
                    res.at[tidx,cidx] = val[1].ub
                else:
                    res.at[tidx,cidx] = val[1].lb
            # set index to time
            res.index = self.pm.timerange[self.set_TIME]
            return res
        else:
            # not implemented
            print('1D and 3D objects not implemented')
            pass
            return None

    def get_objective_value(self):
        return self.OBJ.expr()


    def update_inflow(self):
        """ Update values for inflow parameter """
        m = self.pm
        for a in self.set_HYDRO_AREA:
            if m.opt_hydro_daily:
                for t in self.set_DAYS:
                    self.param_INFLOW[a,t] = pparam_INFLOW(self,a,t)
            else:
                for t in self.set_TIME:
                    self.param_INFLOW[a,t] = pparam_INFLOW(self,a,t)
        self.constr_RESERVOIR_BALANCE.reconstruct()

    def update_ror(self,dic):
        for a,t in dic:
            self.var_HROR[a,t].setub(dic[(a,t)])

    def update_wind(self,dic):
        for a,t in dic:
            self.var_WIND[a,t].setub(dic[(a,t)])

    def update_solar(self,dic):
        for a,t in dic:
            self.var_SOLAR[a,t].setub(dic[(a,t)])


