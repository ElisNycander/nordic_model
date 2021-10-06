"""
Fit cost coefficients for thermal (and hydro) generation
"""


import pickle
import entsoe_transparency_db as entsoe
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_definitions import entsoe_type_map
from help_functions import week_to_date, str_to_date


def f1(x,p,q,nsamp):
    """ x = [k,m1,m2,...,m52]
    args = (p,q,nsamp)
    """
    #    p = args[0] # price
    #    q = args[1] # quantity
    #    nsamp = args[2] # window length
    n = p.__len__() # total number of samples
    m = x.__len__() # number of m coefficients ("bins")
    err = np.zeros([n])
    idx = 0
    ibin = 0
    binidx = 0
    while idx < n:
        # err = p[idx]-k*q[idx]-m[bin[idx]]
        err[idx] = p[idx] - x[0]*q[idx] - x[1+ibin]

        idx += 1
        binidx += 1
        if binidx == nsamp:
            ibin += 1
            binidx = 0
    return err


fig = plt.figure()
fig.set_size_inches(5.5,3.5)

def fit_shifted_costs(starttime='20180101:00',endtime='20181231:23',areas=[],tag='default',
                      binsize=168,nbins=53,loss='linear',period_plots=False,fig_title=True,
                      path = 'D:/NordicModel/Costfits/',db_path='D:/Data',save_eps=True,limit=500,single_plot=True):

    path = Path(path)
    fig_path = path / tag
    fig_path.mkdir(exist_ok=True,parents=True)

    from scipy.optimize import least_squares
    if not areas:
        areas = ['SE1','SE2','SE3','SE4','DK1','DK2','EE','LT','LV', \
                 'FI', 'NO1','NO2','NO3','NO4','NO5']

    entsoe_db = entsoe.Database(db=Path(db_path)/'gen.db')
    gen_data = entsoe_db.select_gen_per_type_wrap_v2(starttime=starttime,endtime=endtime,type_map=entsoe_type_map,
                                                     areas=areas,cet_time=False,dstfix=True,limit=limit)

    price_db = entsoe.Database(db=Path(db_path)/'prices.db')
    price_data = price_db.select_price_data(areas=areas,starttime=starttime,endtime=endtime,cet_time=False)

    # fill nans in price_data
    limit_price = 100
    price_data.fillna(method='ffill',limit=limit_price,inplace=True)
    price_data.fillna(method='bfill',limit=limit_price,inplace=True)
    if price_data.isna().sum().sum() > 0:
        print(f'Too many missing vales in price data: {price_data.isna().sum().sum()}')

    fits = {
        a:{'Thermal':{},'Hydro':{}} for a in areas
    }
    fits['binsize'] = binsize
    fits['starttime'] = starttime

    for area in areas:

        print(f'Fit costs for area {area}')
        for gtype in [f for f in ['Thermal'] if f in gen_data[area].columns]:

            x0 = np.zeros([nbins+1])
            x0[0] = 0.02

            args = (np.array(price_data[area],dtype=float),np.array(gen_data[area][gtype],dtype=float),binsize)

            res = least_squares(f1,x0,args=args,loss=loss)

            if res.success:
                k = res.x[0]
                # calculate "average" m-value, using fitted k-value
                m = np.mean(args[0]) - np.mean(args[1]) * k

                # store values
                fits[area][gtype]['k'] = k
                fits[area][gtype]['m'] = res.x[1:]
                fits[area][gtype]['mavg'] = m

                if single_plot:
                    # create time series
                    year = int(starttime[:4])
                    tindex = pd.date_range(start=str_to_date(starttime),end=str_to_date(endtime),freq='H')

                    mvals = pd.Series(dtype=float,index=tindex)
                    for widx, val in enumerate(res.x[1:]):
                        mvals.at[week_to_date(f'{year}:{str(int(widx+1)).zfill(2)}')] = val
                    mvals.ffill(inplace=True)

                    #%
                    f = plt.gcf()
                    f.clf()
                    f.set_size_inches(6.4,7)
                    ax2,ax1 = f.subplots(2,1,gridspec_kw={'height_ratios':[1,1]})

                    mvals.plot(ax=ax2,label='m (weekly)')
                    tvals = ax2.get_lines()[0].get_xdata()
                    yvals = m*np.ones(tvals.__len__())
                    ax2.plot(tvals,yvals,'--k',label='m (avg)')
                    ax2.grid()
                    ax2.legend()
                    ax2.set_ylabel('m (EUR)')

                    ax1.plot(args[1],args[0],linestyle='None',marker='o',markersize=3,label='data')
                    xvals = np.array([np.min(args[1]),np.max(args[1])])
                    ax1.plot(xvals,np.polyval([k,m],xvals),'r',label='fit')
                    ax1.grid()
                    ax1.legend(title=f'MC=k*P+m\nk={k:.2}, m={m:.3}')
                    ax1.set_ylabel('Price (EUR/MWh)')
                    ax1.set_xlabel('Production (MWh)')

                    plt.savefig(fig_path / f'fit_{area}_{gtype}.png')
                    plt.savefig(fig_path / f'fit_{area}_{gtype}.eps')

                else:
                    # fopt_plots
                    f = plt.gcf()
                    f.clf()
                    # plt.clf()
                    plt.plot(list(range(1,res.x.__len__())),res.x[1:],'-o')
                    plt.grid()
                    plt.xlabel('Week nr.')
                    # plt.xlabel('Period, {0} hours each'.format(args[2]))
                    plt.ylabel('m (EUR)')
                    if fig_title:
                        plt.title('{0} cost fit, MC=k*P+m, k={1:.3}'.format(area,res.x[0]))
                    # plt.gcf().set_size_inches(std_fig_size)
                    plt.tight_layout()
                    plt.savefig(fig_path / f'{tag}_fit_{area}_{gtype}.png')
                    if save_eps:
                        plt.savefig(fig_path / f'{tag}_fit_{area}_{gtype}.eps')

                    # plot scatter plot for whole period
                    # plt.clf()
                    f.clf()
                    plt.plot(args[1]/1e3,args[0],linestyle='None',marker='o',markersize=3,label='data')
                    xvals = np.array([np.min(args[1]),np.max(args[1])])
                    plt.plot(xvals/1e3,np.polyval([k,m],xvals),'r',label='fit')
                    plt.grid()
                    # plt.title(f'{area} yearly cost fit, MC=k*P+m, k={k:.3}, m={m:.3}')
                    plt.legend(title=f'MC=k*P+m\nk={k:.2}, m={m:.3}')
                    plt.ylabel('Price (EUR/MWh)')
                    plt.xlabel('Production (GWh)')
                    plt.tight_layout()
                    plt.savefig(fig_path / f'yearly_fit_{area}_{gtype}.png')
                    if save_eps:
                        plt.savefig(fig_path / f'yearly_fit_{area}_{gtype}.eps')

                if period_plots:
                    fig_path2 = fig_path / f'{area}_{gtype}'
                    fig_path2.mkdir(exist_ok=True,parents=True)
                    for ibin,mx in enumerate(res.x[1:]):
                        if (ibin+1)*args[2] < args[0].__len__():
                            p = args[0][ibin*args[2]:(ibin+1)*args[2]]
                            q = args[1][ibin*args[2]:(ibin+1)*args[2]]
                        else:
                            p = args[0][ibin*args[2]:]
                            q = args[1][ibin*args[2]:]
                        f = plt.gcf()
                        f.set_size_inches(5.5,3.5)
                        f.clf()
                        # plt.clf()
                        plt.plot(q/1e3,p,'o')
                        plt.plot(q/1e3,np.polyval([res.x[0],mx],q),'r')
                        plt.xlabel('Production (GWh)')
                        plt.ylabel('Price (EUR/MWh)')
                        plt.legend(['data','fit'])
                        plt.grid()

                        if fig_title:
                            plt.title('{0} cost fit, bin {1}, k={2:.3}, m={3:.3}'.format(area,ibin,res.x[0],mx))
                        plt.tight_layout()
                        plt.savefig(fig_path2 / f'{tag}_fit_{area}_bin{ibin}.png')
                        if save_eps:
                            plt.savefig(fig_path2 / f'{tag}_fit_{area}_bin{ibin}.eps')

    with open(path / f'{tag}_fit.pkl','wb') as f:
        pickle.dump(fits,f)
    return fits



if __name__ == "__main__":
    pass

    def main():
        pass

