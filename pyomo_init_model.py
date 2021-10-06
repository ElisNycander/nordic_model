import pandas as pd
import pyomo.environ as pye
from pyomo.core.base.PyomoModel import ConcreteModel
from model_definitions import MWtoGW, GWtoTW

# generator limits
def plim_PG(m,igen,itime):
    return (m.pm.min_PG_LR[igen,itime],m.pm.max_PG_LR[igen,itime])

# generation initial value
def pini_PG(m,igen,itime):
    return 0.5*(m.pm.min_PG_LR[(igen,itime)]+m.pm.max_PG_LR[(igen,itime)])

# reservoir storage limits
def plim_RES(m,iarea,itime):
    return (0,m.pm.max_RES_LR[(iarea,itime)])


def plim_HROR(m,iarea,itime):
    return (0,m.pm.max_HROR_LR[(iarea,itime)])

def pini_HROR(m,iarea,itime):
    return m.pm.max_HROR_LR[(iarea,itime)]

def plim_PRES(m,iarea,itime):
    return (0,m.pm.max_PRES[(iarea,itime)])

def plim_PUMP(m,iarea,itime):
    return (0,m.pm.max_PUMP[(iarea,itime)])

def plim_X1(m,ixtrans,itime):
    return (0,m.pm.max_X1_LR[(ixtrans,itime)])

def plim_X2(m,ixtrans,itime):
    return (0,m.pm.max_X2_LR[(ixtrans,itime)])

# limits for variable external transmissions
def plim_XEXT(m,ixtrans,itime):
    return (m.pm.min_XEXT_LR[(ixtrans,itime)],m.pm.max_XEXT[(ixtrans,itime)])

# external transmissions
def pparam_X(m,ixtrans,itime):
    return m.pm.exchange_lr.at[m.pm.timerange_lr[itime],m.pm.xtrans_ext.at[ixtrans,'label_fw']] * MWtoGW

# hydro inflow
def pparam_INFLOW(m,ihydro,itime):
    return m.pm.inflow_hourly_lr.at[m.pm.timerange_lr[itime],ihydro]

def plim_LS(m,iarea,itime):
    return (0,m.pm.max_LS_LR[(iarea,itime)])

def plim_W(m,iarea,itime):
    return (0,m.pm.max_WIND_LR[(iarea,itime)])

def pini_W(m,iarea,itime):
    return m.pm.max_WIND_LR[(iarea,itime)]

def plim_SOLAR(m,iarea,itime):
    return (0,m.pm.max_SOLAR_LR[(iarea,itime)])

def pini_SOLAR(m,iarea,itime):
    return m.pm.max_SOLAR_LR[(iarea,itime)]

def pini_RES(m,iarea,itime):
    return m.pm.reservoir_interp_lr.at[m.pm.timerange_lr[itime],iarea]


def pconstr_POWER_BALANCE(m,area,t):
    #    if area in m.set_ROR_AREA: # has run of river
    balance = sum( m.var_PG[g,t] for g in m.set_AREA_TO_GEN[area]) \
              - sum( m.var_X1[c,t] for c in m.set_XINT_FW[area] ) \
              - sum( m.var_X2[c,t] for c in m.set_XINT_BW[area] ) \
              + (1-m.pm.opt_loss) * sum( m.var_X1[c,t] for c in m.set_XINT_BW[area] ) \
              + (1-m.pm.opt_loss) * sum( m.var_X2[c,t] for c in m.set_XINT_FW[area] ) \
              - MWtoGW * sum( m.pm.exchange_lr.at[m.pm.timerange_lr[t],m.pm.xtrans_ext.at[c,'label_fw']]
                              for c in m.set_AREA_TO_XEXT_PAR[area]) \
              - sum( m.var_XEXT[c,t] for c in m.set_AREA_TO_XEXT_VAR[area]) + \
              - m.pm.max_LS_LR[(area,t)] + m.var_LS[area,t]
    if area in m.set_WIND_AREA:
        balance += m.var_WIND[area,t]
    if area in m.set_SOLAR_AREA:
        balance += m.var_SOLAR[area,t]
    if area in m.set_ROR_AREA:
        balance += m.var_HROR[area,t]
    return balance == 0

def pconstr_RESERVOIR_BALANCE(m,a,t):
    scale = GWtoTW
    # final reservoir value
    expr = -m.var_RES[a,t]
    # initial reservoir value
    if t == 0: # first period
        expr += m.pm.reservoir_fix.at[ m.pm.timerange_lr[0],a]
    else:
        expr += m.var_RES[a,t-1]
    # add inflow
    if a in m.set_ROR_AREA: # subtract ror from inflow
        expr += scale*m.pm.opt_init_delta*max(0,m.pm.inflow_hourly_lr.at[m.pm.timerange_lr[t],a]-MWtoGW*m.pm.ror_hourly_lr.at[m.pm.timerange_lr[t],a])
    else:
        expr += scale*m.pm.opt_init_delta*m.pm.inflow_hourly_lr.at[m.pm.timerange_lr[t],a]
    # subtract production
    expr -= scale*sum(m.var_PG[g,t] for g in m.set_HYDRO_AREA_TO_GEN[a])*m.pm.opt_init_delta
    # subtract spillage
    expr -= scale*m.var_SPILLAGE[a,t]*m.pm.opt_init_delta
    return expr == 0

# Note: For quadratic costs, appropriate scaling of cost coefficients is needed
# m.pm.gen_data has EUR/MW, but m has generation in GW
# [c1*var_PG/MWtoGW] = EUR
# [c2*var_PG^2/MWtoGW^2] = EUR
# [ext_price*var_XEXT/MWtoGW] = EUR
# [opt_loadshed_cost*var_D/MWtoGW] = EUR
# Scale down costs by 10e6: * MWtoGW^2, so we use:
# c1*var_PG*MWtoGW = EUR/10e6
# c2*var_PG^2 = EUR/10e6
# ext_price*var_XEXT*MWtoGW = EUR/10e6
# opt_loadshed_cost*var_D*MWtoGW = EUR/10e6


# thermal_cost_rule - only thermal costs
def pobj_THERMAL_COST(m): # thermal costs
    if m.pm.opt_use_var_cost:
        val = sum( sum(m.pm.gen_c2_lr.at[m.pm.timerange_lr[t],g]*m.var_PG[g,t]**2 + m.pm.gen_c1_lr.at[m.pm.timerange_lr[t],g]*MWtoGW*m.var_PG[g,t]  for t in m.set_TIME) for g in m.set_THERMAL_GEN) \
              - sum( sum(m.pm.price_external.at[m.pm.timerange_lr[t],m.pm.xtrans_ext.at[x,'to']]*MWtoGW*m.var_XEXT[x,t] for t in m.set_TIME) for x in m.set_XEXT_VAR ) \
              + sum( sum( m.pm.opt_loadshed_cost*MWtoGW*m.var_LS[a,t] for t in m.set_TIME) for a in m.set_AREA) \
              + sum( sum( m.var_WIND[a,t]*m.pm.opt_wind_cost*MWtoGW for t in m.set_TIME) for a in m.set_WIND_AREA)
        if m.pm.opt_hydro_cost:
            val += sum( m.pm.gen_c2_lr.at[m.pm.timerange_lr[t],g]*m.var_PG[g,t]**2 + m.pm.gen_c1_lr.at[m.pm.timerange_lr[t],g]*MWtoGW*m.var_PG[g,t] for t in m.set_TIME for g in m.set_HYDRO_GEN)
    else:
        val = sum( sum(m.pm.gen_data.at[g,'c2']*m.var_PG[g,t]**2 + m.pm.gen_data.at[g,'c1']*MWtoGW*m.var_PG[g,t]  for t in m.set_TIME) for g in m.set_THERMAL_GEN) \
              - sum( sum(m.pm.price_external.at[m.pm.timerange_lr[t],m.pm.xtrans_ext.at[x,'to']]*MWtoGW*m.var_XEXT[x,t] for t in m.set_TIME) for x in m.set_XEXT_VAR ) \
              + sum( sum( m.pm.opt_loadshed_cost*MWtoGW*m.var_LS[a,t] for t in m.set_TIME) for a in m.set_AREA) \
              + sum( sum( m.var_WIND[a,t]*m.pm.opt_wind_cost*MWtoGW for t in m.set_TIME) for a in m.set_WIND_AREA)

    return val*m.pm.opt_init_delta


def pconstr_HVDC_RAMP(m,c,t):
    if t == 0:
        return pye.Constraint.Skip
    else:
        return pye.inequality(-m.pm.opt_hvdc_max_ramp*MWtoGW, sum((m.var_X1[conn,t]-m.var_X2[conn,t])-(m.var_X1[conn,t-1]-m.var_X2[conn,t-1]) for conn in m.set_HVDC_TO_XINT[c]),m.pm.opt_hvdc_max_ramp*MWtoGW)

def pconstr_HVDC_RAMP_EXT(m,c,t):
    if t == 0:
        return pye.Constraint.Skip
    else:
        return pye.inequality(-m.pm.xtrans_ext.at[c,'ramp']*MWtoGW, m.var_XEXT[c,t]-m.var_XEXT[c,t-1], m.pm.xtrans_ext.at[c,'ramp']*MWtoGW)

def pconstr_PG_RAMP(m,gen,t):
    if t == 0:
        return pye.Constraint.Skip
    else:
        # sum over combined generators
        return pye.inequality(m.pm.gen_data_comb.at[gen,'rampdown']*MWtoGW,sum(m.var_PG[g,t]-m.var_PG[g,t-1] for g in m.set_COMB_GEN_TO_GEN[gen]),m.pm.gen_data_comb.at[gen,'rampup']*MWtoGW)

def pconstr_FINAL_RESERVOIR(m,area):
    tN = max(list(set(m.set_TIME)))
    return m.var_RES[area,tN] == m.pm.reservoir_fix.at[m.pm.timerange_p1[-1],area]

# Note: with country reserves and ror, the ror from all areas is included in the generation, irrespective of which reservoir hydro generators are actually included
def pconstr_RESERVES_UP(m,area,t):
    if m.pm.opt_country_reserves:
        if area in m.set_ROR_COUNTRY:
            return sum(m.pm.gen_data.at[g,'pmax']*MWtoGW - m.var_PG[g,t] for g in m.set_RESERVE_COUNTRY_TO_GEN[area]) - sum(m.var_HROR[a,t] for a in m.pm.country_to_areas[area]) >= m.pm.reserve_country_data.at[area,'Rp']*MWtoGW
        else:
            return sum(m.pm.gen_data.at[g,'pmax']*MWtoGW - m.var_PG[g,t] for g in m.set_RESERVE_COUNTRY_TO_GEN[area]) >= m.pm.reserve_country_data.at[area,'Rp']*MWtoGW
    else:
        # check if there is hydro generator providing reserves
        if area in m.set_ROR_AREA:
            return sum(m.pm.gen_data.at[g,'pmax']*MWtoGW - m.var_PG[g,t] for g in m.set_RESERVE_AREA_TO_GEN[area]) - m.var_HROR[area,t] >= m.pm.reserve_data.at[area,'Rp']*MWtoGW
        else:
            return sum(m.pm.gen_data.at[g,'pmax']*MWtoGW - m.var_PG[g,t] for g in m.set_RESERVE_AREA_TO_GEN[area]) >= m.pm.reserve_data.at[area,'Rp']*MWtoGW

def pconstr_RESERVES_DW(m,area,t):
    if m.pm.opt_country_reserves:
        if area in m.set_ROR_COUNTRY:
            return sum(m.var_PG[g,t] - m.pm.gen_data.at[g,'pmin']*MWtoGW for g in m.set_RESERVE_COUNTRY_TO_GEN[area]) + sum(m.var_HROR[a,t] for a in m.pm.country_to_areas[area]) >= m.pm.reserve_country_data.at[area,'Rn']*MWtoGW
        else:
            return sum(m.var_PG[g,t] - m.pm.gen_data.at[g,'pmin']*MWtoGW for g in m.set_RESERVE_COUNTRY_TO_GEN[area]) >= m.pm.reserve_country_data.at[area,'Rn']*MWtoGW
    else:
        if area in m.set_ROR_AREA:
            return sum(m.var_PG[g,t] - m.pm.gen_data.at[g,'pmin']*MWtoGW for g in m.set_RESERVE_AREA_TO_GEN[area]) + m.var_HROR[area,t] >= m.pm.reserve_data.at[area,'Rn']*MWtoGW
        else:
            return sum(m.var_PG[g,t] - m.pm.gen_data.at[g,'pmin']*MWtoGW for g in m.set_RESERVE_AREA_TO_GEN[area]) >= m.pm.reserve_data.at[area,'Rn']*MWtoGW

def pconstr_INERTIA(m,t):
    # Note: should include run of river
    return sum(m.var_PG[g,t] \
               * m.pm.opt_inertia_constants[m.pm.area_to_country[m.pm.gen_data.at[g,'area']]][m.pm.gen_data.at[g,'gtype']] \
               / m.pm.opt_inertia_cf[m.pm.area_to_country[m.pm.gen_data.at[g,'area']]][m.pm.gen_data.at[g,'gtype']] \
               / m.pm.opt_inertia_pf[m.pm.area_to_country[m.pm.gen_data.at[g,'area']]][m.pm.gen_data.at[g,'gtype']] \
               for g in m.set_GEN if m.pm.gen_data.at[g,'area'] in m.set_SYNCAREA) + \
           sum(m.var_HROR[a,t] * m.opt_inertia_constants[m.pm.area_to_country[a]]['Hydro'] \
               / m.pm.opt_inertia_cf[m.pm.area_to_country[a]]['Hydro'] \
               / m.pm.opt_inertia_pf[m.pm.area_to_country[a]]['Hydro'] \
               for a in m.set_ROR_AREA if a in m.set_SYNCAREA) \
           >= m.pm.opt_min_kinetic_energy

def pconstr_MIN_HYDRO(m,a,t):
    hgens = m.set_HYDRO_AREA_TO_GEN[a]
    hmin = m.pm.gen_data.loc[hgens,'pmin'].sum() * MWtoGW
    if a in m.set_ROR_AREA:
        return sum( m.var_PG[g,t] for g in hgens) + m.var_HROR[a,t] >= hmin
    else:
        return sum( m.var_PG[g,t] for g in hgens) >= hmin

def pconstr_MAX_HYDRO(m,a,t):
    hgens = m.set_HYDRO_AREA_TO_GEN[a]
    hmax = m.pm.gen_data.loc[hgens,'pmax'].sum() * MWtoGW
    if a in m.set_ROR_AREA:
        return sum( m.var_PG[g,t] for g in hgens) + m.var_HROR[a,t] <= hmax
    else:
        return sum( m.var_PG[g,t] for g in hgens) <= hmax


class PyomoInitModel(ConcreteModel):
    """ Wrapper class pyomo
    All parts of the modelling related to the api of the pyomo model goes here
    """
    def __init__(self):
        # call ConcreteModel constructor
        super(PyomoInitModel,self).__init__()
        self.print_flag = False

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
        self.print_flag = m.opt_print['init']
        print_output = self.print_flag

        ## SETS
        self.set_TIME = pye.Set(initialize=range(m.nPeriods_lr))
        self.set_AREA = pye.Set(initialize=m.areas)
        self.set_COUNTRY = pye.Set(initialize=m.opt_countries)
        self.set_C2A = pye.Set(self.set_COUNTRY,within=self.set_AREA,initialize=m.country_to_areas)
        self.set_SYNCAREA = pye.Set(initialize=m.syncareas)
        self.set_ROR_AREA = pye.Set(initialize=m.ror_areas)
        self.set_ROR_COUNTRY = pye.Set(initialize=m.ror_countries)

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
        self.set_PUMP_AREA = pye.Set(initialize=m.pump_areas)
        self.set_PUMP_RES_AREA = pye.Set(initialize=m.pump_res_areas)
        self.set_PUMP_NORES_AREA = pye.Set(initialize=m.pump_nores_areas)
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
        self.set_AREA_TO_XEXT_PAR = pye.Set(self.set_AREA,within=self.set_XEXT,initialize=m.xext_ft)
        self.set_AREA_TO_XEXT_VAR = pye.Set(self.set_AREA,within=self.set_XEXT,initialize=m.xext_fp)

        self.set_HVDC = pye.Set(initialize = [i for i in m.combined_hvdc.keys()])
        self.set_HVDC_TO_XINT = pye.Set(self.set_HVDC,within=self.set_XINT,initialize = m.combined_hvdc)

        ## VARIABLES
        if print_output:
            print('Setting up VARIABLES')

        # load shedding
        self.var_LS = pye.Var(self.set_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_LS,initialize=0)
        ## all generators
        # production
        self.var_PG = pye.Var(self.set_GEN,self.set_TIME,domain=pye.Reals,bounds=plim_PG,initialize=pini_PG)
        # wind generation
        self.var_WIND = pye.Var(self.set_WIND_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_W,initialize=pini_W)
        # solar generation
        self.var_SOLAR = pye.Var(self.set_SOLAR_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_SOLAR,initialize=pini_SOLAR)
        ## hydro generators
        # spillage
        self.var_SPILLAGE = pye.Var(self.set_HYDRO_AREA,self.set_TIME,domain=pye.Reals,bounds=(0,None),initialize=0)
        # reservoir storage
        self.var_RES = pye.Var(self.set_HYDRO_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_RES,initialize=pini_RES)
        # run of river hydro
        self.var_HROR = pye.Var(self.set_ROR_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_HROR,initialize=pini_HROR)
        self.var_PUMP = pye.Var(self.set_PUMP_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_PUMP,initialize=0)
        self.var_REL = pye.Var(self.set_PUMP_RES_AREA,self.set_TIME,domain=pye.Reals,bounds=(0,None),initialize=0)
        self.var_PRES = pye.Var(self.set_PUMP_RES_AREA,self.set_TIME,domain=pye.Reals,bounds=plim_PRES,initialize=0)
        ## internal transmission
        self.var_X1 = pye.Var(self.set_XINT,self.set_TIME,domain=pye.Reals,bounds=plim_X1,initialize=0)
        self.var_X2 = pye.Var(self.set_XINT,self.set_TIME,domain=pye.Reals,bounds=plim_X2,initialize=0)
        # external transmission to areas with fixed price
        self.var_XEXT = pye.Var(self.set_XEXT_VAR,self.set_TIME,domain=pye.Reals,bounds=plim_XEXT,initialize=0)

        ## OBJECTIVE

        self.OBJ = pye.Objective(rule=pobj_THERMAL_COST)

        ## CONSTRAINTS
        if print_output:
            print('Setting up CONSTRAINTS')
        self.constr_POWER_BALANCE = pye.Constraint(self.set_AREA,self.set_TIME,rule=pconstr_POWER_BALANCE)
        #
        self.constr_RESERVOIR_BALANCE = pye.Constraint(self.set_HYDRO_AREA,self.set_TIME,rule=pconstr_RESERVOIR_BALANCE)
        self.constr_FIX_RESERVOIR = pye.Constraint(self.set_HYDRO_AREA,rule=pconstr_FINAL_RESERVOIR)

        self.constr_MIN_HYDRO = pye.Constraint(self.set_HYDRO_AREA,self.set_TIME,rule=pconstr_MIN_HYDRO)
        self.constr_MAX_HYDRO = pye.Constraint(self.set_HYDRO_AREA,self.set_TIME,rule=pconstr_MAX_HYDRO)

        self.constr_HVDC_RAMP = pye.Constraint(self.set_HVDC,self.set_TIME,rule=pconstr_HVDC_RAMP)
        # #
        self.constr_GEN_RAMP = pye.Constraint(self.set_COMB_GEN,self.set_TIME,rule=pconstr_PG_RAMP)
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
            res.index = self.pm.timerange_lr[self.set_TIME]
            return res
        else:
            # not implemented
            print('1D and 3D objects not implemented')
            pass
            return None

    def get_objective_value(self):
        return self.OBJ.expr()

