""" Model rules used to setup model with Pyomo

names:
pconstr_XX - pyomo constraint rule
plim_XX - pyomo limit rule for variable
pobj_XX - pyomo objective rule
pini_XX - pyomo initial value rule for variable
pparam_XX - pyomo parameter rule
"""

from model_definitions import MWtoGW, GWtoTW
import pyomo.environ as pye

# generator limits
def plim_PG(m,igen,itime):
    return (m.pm.min_PG[igen,itime],m.pm.max_PG[igen,itime])

# generation initial value
def pini_PG(m,igen,itime):
    if hasattr(m.pm,'ini'):
        return m.pm.ini.res_PG.at[m.pm.timerange_lr[m.pm.i2p[itime]],igen]
    else:
        return 0.5*(m.pm.min_PG[(igen,itime)]+m.pm.max_PG[(igen,itime)])

# reservoir storage limits
def plim_RES(m,iarea,itime):
    return (0,m.pm.reservoir_capacity[iarea])

def plim_HROR(m,iarea,itime):
    return (0,m.pm.max_HROR[(iarea,itime)])

def pini_HROR(m,iarea,itime):
    if hasattr(m.pm,'ini'):
        # return m.pm.ini.res_HROR.at[m.pm.timerange_lr[m.pm.i2p[itime]],iarea]
        return m.pm.max_HROR[(iarea,itime)]
    else:
        return m.pm.max_HROR[(iarea,itime)]

def plim_PUMP(m,iarea,itime):
    return (0,m.pm.max_PUMP[(iarea,itime)])

def pini_PUMP(m,iarea,itime):
    if hasattr(m.pm,'ini'):
        return 0
    else:
        return 0

def plim_PRES(m,iarea,itime):
    return (0,m.pm.pump_reservoir[iarea])

def plim_REL(m,iarea,itime):
    return (0,None)

def plim_X1(m,ixtrans,itime):
    return (0,m.pm.max_X1[(ixtrans,itime)])

def pini_X1(m,ixtrans,itime):
    if hasattr(m.pm,'ini'):
        return m.pm.ini.res_X1.at[m.pm.timerange_lr[m.pm.i2p[itime]],ixtrans]
    else:
        return 0

def plim_X2(m,ixtrans,itime):
    return (0,m.pm.max_X2[(ixtrans,itime)])

def pini_X2(m,ixtrans,itime):
    if hasattr(m.pm,'ini'):
        return m.pm.ini.res_X2.at[m.pm.timerange_lr[m.pm.i2p[itime]],ixtrans]
    else:
        return 0
# limits for variable external transmissions
def plim_XEXT(m,ixtrans,itime):
    return (m.pm.min_XEXT[(ixtrans,itime)],m.pm.max_XEXT[(ixtrans,itime)])

def pini_XEXT(m,ixtrans,itime):
    if hasattr(m.pm,'ini'):
        return m.pm.ini.res_XEXT.at[m.pm.timerange_lr[m.pm.i2p[itime]],ixtrans]
    else:
        return 0

# external transmissions
def pparam_X(m,ixtrans,itime):
    return m.pm.exchange.at[m.pm.timerange[itime],m.pm.xtrans_ext.at[ixtrans,'label_fw']] * MWtoGW

# hydro reservoir starting values
def pini_M(m,ihydro):
    return m.pm.reservoir_hourly.at[ m.pm.timerange[0], ihydro ]

# hydro inflow
def pparam_INFLOW(m,a,t):
    if m.pm.opt_hydro_daily:
        if a in m.set_ROR_AREA: # subtract ror from inflow
            return max(0,m.pm.inflow_daily.at[m.pm.daysrange[t],a]-MWtoGW*m.pm.ror_daily.at[m.pm.daysrange[t],a])
        else:
            return m.pm.inflow_daily.at[m.pm.daysrange[t],a]
    else:
        if a in m.set_ROR_AREA: # subtract ror from inflow
            return max(0,m.pm.inflow_hourly.at[m.pm.timerange[t],a]-MWtoGW*m.pm.ror_hourly.at[m.pm.timerange[t],a])
        else:
            return m.pm.inflow_hourly.at[m.pm.timerange[t],a]

# demand for area (uncurtailed demand)
def pini_LS(m,iarea,itime):
    if hasattr(m.pm,'ini'):
        # return m.pm.ini.res_LS.at[m.pm.timerange_lr[m.pm.i2p[itime]],iarea]
        return 0
    else:
        # return m.pm.max_LS[(iarea,itime)]
        return 0

def plim_LS(m,iarea,itime):
    return (0,m.pm.max_LS[(iarea,itime)])

def plim_W(m,iarea,itime):
    return (0,m.pm.max_WIND[(iarea,itime)])

def pini_W(m,iarea,itime):
    if hasattr(m.pm,'ini'):
        # return m.pm.ini.res_WIND.at[m.pm.timerange_lr[m.pm.i2p[itime]],iarea]
        return m.pm.max_WIND[(iarea,itime)]
    else:
        return m.pm.max_WIND[(iarea,itime)]

def plim_SOLAR(m,iarea,itime):
    return (0,m.pm.max_SOLAR[(iarea,itime)])

def pini_SOLAR(m,iarea,itime):
    if hasattr(m.pm,'ini'):
        # return m.pm.ini.res_SOLAR.at[m.pm.timerange_lr[m.pm.i2p[itime]],iarea]
        return m.pm.max_SOLAR[(iarea,itime)]
    else:
        return m.pm.max_SOLAR[(iarea,itime)]

def pini_RES(m,iarea,itime):
    if hasattr(m.pm,'ini'):
        if m.pm.opt_hydro_daily:
            return m.pm.ini.reservoir_interp.at[m.pm.daysrange[itime],iarea]
        else:
            return m.pm.ini.reservoir_interp.at[m.pm.timerange[itime],iarea]
    else:
        if m.pm.opt_hydro_daily:
            return m.pm.reservoir_interp.at[m.pm.daysrange[itime],iarea]
        else:
            return m.pm.reservoir_interp.at[m.pm.timerange[itime],iarea]

def pconstr_PUMP_BALANCE(m,a,t):
    scale = GWtoTW
    if t == m.set_TIME[1]:
        pres_tm1 = 0
    else:
        pres_tm1 = m.var_PRES[a,t-1]
    return m.var_PRES[a,t] == pres_tm1 + scale*(m.pm.opt_pump_efficiency*m.var_PUMP[a,t] - m.var_REL[a,t])

def pconstr_POWER_BALANCE(m,area,t):
    #    if area in m.set_ROR_AREA: # has run of river
    balance = sum( m.var_PG[g,t] for g in m.set_AREA_TO_GEN[area]) \
              - sum( m.var_X1[c,t] for c in m.set_XINT_FW[area] ) \
              - sum( m.var_X2[c,t] for c in m.set_XINT_BW[area] ) \
              + (1-m.pm.opt_loss) * sum( m.var_X1[c,t] for c in m.set_XINT_BW[area] ) \
              + (1-m.pm.opt_loss) * sum( m.var_X2[c,t] for c in m.set_XINT_FW[area] ) \
              - MWtoGW * sum( m.pm.exchange.at[m.pm.timerange[t],m.pm.xtrans_ext.at[c,'label_fw']]
                              for c in m.set_AREA_TO_XEXT_PAR[area]) \
              - sum( m.var_XEXT[c,t] for c in m.set_AREA_TO_XEXT_VAR[area]) + \
              - m.pm.max_LS[(area,t)] + m.var_LS[area,t]
    if area in m.set_WIND_AREA:
        balance += m.var_WIND[area,t]
    if area in m.set_SOLAR_AREA:
        balance += m.var_SOLAR[area,t]
    if area in m.set_ROR_AREA:
        balance += m.var_HROR[area,t]
    if area in m.set_PUMP_AREA:
        balance -= m.var_PUMP[area,t]
    if area in m.set_PUMP_RES_AREA:
        balance += m.var_REL[area,t]

    return balance == 0

def pconstr_RESERVOIR_BALANCE(m,a,t):
    scale = GWtoTW
    # final reservoir value
    expr = -m.var_RES[a,t]
    # initial reservoir value
    if t == 0: # first period
        expr += m.pm.reservoir_fix.at[ m.pm.timerange[0],a]
    else:
        expr += m.var_RES[a,t-1]
    # add inflow
    expr += scale*m.param_INFLOW[a,t]
    # if a in m.set_ROR_AREA: # subtract ror from inflow
    #     expr += scale*max(0,m.pm.inflow_hourly.at[m.pm.timerange[t],a]-MWtoGW*m.pm.ror_hourly.at[m.pm.timerange[t],a])
    # else:
    #     expr += scale*m.pm.inflow_hourly.at[m.pm.timerange[t],a]
    # subtract production
    expr -= scale*sum(m.var_PG[g,t] for g in m.set_HYDRO_AREA_TO_GEN[a])
    # subtract spillage
    expr -= scale*m.var_SPILLAGE[a,t]
    # add pumping to main reservoir:
    if a in m.set_PUMP_NORES_AREA:
        expr += scale*m.var_PUMP[a,t]*m.pm.opt_pump_efficiency
    return expr == 0

def pconstr_RESERVOIR_BALANCE_DAILY(m,a,t):
    scale = GWtoTW
    # final reservoir value
    expr = -m.var_RES[a,t]
    # initial reservoir value
    if t == 0: # first period
        expr += m.pm.reservoir_fix.at[ m.pm.daysrange[0],a]
    else:
        expr += m.var_RES[a,t-1]
    # add inflow
    if a in m.set_ROR_AREA: # subtract ror from inflow
        expr += scale*max(0,m.pm.inflow_daily.at[m.pm.daysrange[t],a]-MWtoGW*m.pm.ror_daily.at[m.pm.daysrange[t],a])
    else:
        expr += scale*m.pm.inflow_daily.at[m.pm.daysrange[t],a]
    # subtract production
    expr -= scale*sum(m.var_PG[g,tt] for g in m.set_HYDRO_AREA_TO_GEN[a] for tt in m.pm.day2hour[t])
    # subtract spillage
    expr -= scale*m.var_SPILLAGE[a,t]
    # add pumping into main reservoir
    if a in m.set_PUMP_NORES_AREA:
        expr += scale*sum(m.var_PUMP[a,tt] for tt in m.pm.day2hour[t])*m.pm.opt_pump_efficiency
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
        val = sum( sum(m.pm.gen_c2.at[m.pm.timerange[t],g]*m.var_PG[g,t]**2 + m.pm.gen_c1.at[m.pm.timerange[t],g]*MWtoGW*m.var_PG[g,t]  for t in m.set_TIME) for g in m.set_THERMAL_GEN) \
              - sum( sum(m.pm.price_external.at[m.pm.timerange[t],m.pm.xtrans_ext.at[x,'to']]*MWtoGW*m.var_XEXT[x,t] for t in m.set_TIME) for x in m.set_XEXT_VAR ) \
              + sum( sum( m.pm.opt_loadshed_cost*MWtoGW*m.var_LS[a,t] for t in m.set_TIME) for a in m.set_AREA) \
              + sum( sum( m.var_WIND[a,t]*m.pm.opt_wind_cost*MWtoGW for t in m.set_TIME) for a in m.set_WIND_AREA)
        if m.pm.opt_hydro_cost:
            val += sum( m.pm.gen_c2.at[m.pm.timerange[t],g]*m.var_PG[g,t]**2 + m.pm.gen_c1.at[m.pm.timerange[t],g]*MWtoGW*m.var_PG[g,t] for t in m.set_TIME for g in m.set_HYDRO_GEN)
    else:
        val = sum( sum(m.pm.gen_data.at[g,'c2']*m.var_PG[g,t]**2 + m.pm.gen_data.at[g,'c1']*MWtoGW*m.var_PG[g,t]  for t in m.set_TIME) for g in m.set_THERMAL_GEN) \
              - sum( sum(m.pm.price_external.at[m.pm.timerange[t],m.pm.xtrans_ext.at[x,'to']]*MWtoGW*m.var_XEXT[x,t] for t in m.set_TIME) for x in m.set_XEXT_VAR ) \
              + sum( sum( m.pm.opt_loadshed_cost*MWtoGW*m.var_LS[a,t] for t in m.set_TIME) for a in m.set_AREA) \
              + sum( sum( m.var_WIND[a,t]*m.pm.opt_wind_cost*MWtoGW for t in m.set_TIME) for a in m.set_WIND_AREA)

    return val


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

def pconstr_FINAL_RESERVOIR_DAILY(m,area):
    tN = max(list(set(m.set_DAYS)))
    return m.var_RES[area,tN] == m.pm.reservoir_fix.at[m.pm.daysrange_p1[-1],area]

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
    return sum(m.var_PG[g,t] \
               * m.pm.opt_inertia_constants[m.pm.area_to_country[m.pm.gen_data.at[g,'area']]][m.pm.gen_data.at[g,'gtype']] \
               / m.pm.opt_inertia_cf[m.pm.area_to_country[m.pm.gen_data.at[g,'area']]][m.pm.gen_data.at[g,'gtype']] \
               / m.pm.opt_inertia_pf[m.pm.area_to_country[m.pm.gen_data.at[g,'area']]][m.pm.gen_data.at[g,'gtype']] \
               for g in m.set_GEN if m.pm.gen_data.at[g,'area'] in m.set_SYNCAREA) + \
           sum(m.var_HROR[a,t] * m.pm.opt_inertia_constants[m.pm.area_to_country[a]]['Hydro'] \
               / m.pm.opt_inertia_cf[m.pm.area_to_country[a]]['Hydro'] \
               / m.pm.opt_inertia_pf[m.pm.area_to_country[a]]['Hydro'] \
               for a in m.set_ROR_AREA if a in m.set_SYNCAREA) + \
           sum(m.var_REL[a,t] * m.pm.opt_inertia_constants[m.pm.area_to_country[a]]['Hydro'] \
               / m.pm.opt_inertia_cf[m.pm.area_to_country[a]]['Hydro'] \
               / m.pm.opt_inertia_pf[m.pm.area_to_country[a]]['Hydro'] \
               for a in m.set_PUMP_RES_AREA if a in m.set_SYNCAREA) \
           >= m.pm.opt_min_kinetic_energy

def pconstr_MIN_HYDRO(m,a,t):
    hgens = m.set_HYDRO_AREA_TO_GEN[a]
    hmin = m.pm.gen_data.loc[hgens,'pmin'].sum() * MWtoGW
    if a in m.set_ROR_AREA:
        if a in m.set_PUMP_RES_AREA:
            return sum( m.var_PG[g,t] for g in hgens) + m.var_HROR[a,t] + m.var_REL[a,t] >= hmin
        else:
            return sum( m.var_PG[g,t] for g in hgens) + m.var_HROR[a,t] >= hmin
    else:
        if a in m.set_PUMP_RES_AREA:
            return sum( m.var_PG[g,t] for g in hgens) + m.var_REL[a,t] >= hmin
        else:
            return sum( m.var_PG[g,t] for g in hgens) >= hmin

def pconstr_HYDRO_RAMP(m,a,t):

    if t == 0:
        return pye.Constraint.Skip
    else:
        # sum over combined generators
        hgens = [g for g in m.set_GEN if m.pm.gen_data.at[g,'area'] == a and \
                     m.pm.gen_data.at[g,'gtype'] == 'Hydro']
        max_ramp = m.pm.gen_data.loc[hgens,'rampup'].sum()
        min_ramp = m.pm.gen_data.loc[hgens,'rampdown'].sum()
        expr = sum(m.var_PG[g,t] - m.var_PG[g,t-1] for g in hgens)
        if a in m.set_PUMP_RES_AREA:
            expr += m.var_REL[a,t] - m.var_REL[a,t-1]
        if a in m.set_ROR_AREA:
            expr += m.var_HROR[a,t] - m.var_HROR[a,t-1]

        return pye.inequality(min_ramp*MWtoGW,expr,max_ramp*MWtoGW)
        # return pye.inequality(m.pm.gen_data_comb.at[gen,'rampdown']*MWtoGW,sum(m.var_PG[g,t]-m.var_PG[g,t-1] for g in m.set_COMB_GEN_TO_GEN[gen]),m.pm.gen_data_comb.at[gen,'rampup']*MWtoGW)

def pconstr_THERMAL_RAMP(m,g,t):
    if t == 0:
        return pye.Constraint.Skip
    else:
        return pye.inequality(m.pm.gen_data.at[g,'rampdown']*MWtoGW,
                              m.var_PG[g,t]-m.var_PG[g,t-1],
                              m.pm.gen_data.at[g,'rampup']*MWtoGW)


def pconstr_MAX_HYDRO(m,a,t):
    hgens = m.set_HYDRO_AREA_TO_GEN[a]
    hmax = m.pm.gen_data.loc[hgens,'pmax'].sum() * MWtoGW
    if a in m.set_ROR_AREA:
        if a in m.set_PUMP_RES_AREA:
            return sum( m.var_PG[g,t] for g in hgens) + m.var_HROR[a,t] + m.var_REL[a,t] <= hmax
        else:
            return sum( m.var_PG[g,t] for g in hgens) + m.var_HROR[a,t] <= hmax
    else:
        if a in m.set_PUMP_RES_AREA:
            return sum( m.var_PG[g,t] for g in hgens) + m.var_REL[a,t] <= hmax
        else:
            return sum( m.var_PG[g,t] for g in hgens) <= hmax

if __name__ == "__main__":

    pass