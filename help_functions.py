# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:03:38 2019

@author: elisn

This module has various help functions for model

For conversion between dates (YYYYMMDD:HH) and weeks (YYYY:WW) weeks are counted as starting during the first hour
in a year and lasting 7 days, except for the last week which covers the remaining hours in the year. Thus all years
are assumed to have 52 weeks. This definition is not according to ISO calendar standard but is legacy from the
first version of the model, probably changing it would not significantly change the results. Also note that the
MAF inflow data used also does not follow ISO calendar standard for weeks but counts weeks as starting with Sundays.
"""
import numpy as np
import datetime
import pandas as pd
import re

weekdd = ['0{0}'.format(i) for i in range(1,10)] + [str(i) for i in range(10,53)]

seconds_per_hour = 3600
hours_per_day = 24


def compute_df_rmse(df1,df2,power=1):
    if power == 1:
        return (df1-df2).abs().mean()
    else:
        return ((df1-df2)**2).mean().apply(np.sqrt)


# Polynomial Regression
def polyfit(x, y, degree=1):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

def replace_outliers(series,minpc=10,maxpc=90):
    """
    Truncate distribution at given percentiles
    :param series: 
    :param minpc: 
    :param maxpc: 
    :return: 
    """
    vals = np.percentile(np.array(series),[minpc,maxpc])
    series.loc[series < vals[0]] = vals[0]
    series.loc[series > vals[1]] = vals[1]
    return series

def score_name(name1,name2,aao = False):

    """ Hur lika är name1 de olika elementen i name2?
        Returnerar array name1 * name2
        aao = True -> ersätter Å, Ä, Ö med A, A, O"""

    def swe(s):
        s = s.replace('Å','A')
        s = s.replace('Ä','A')
        s = s.replace('Ö','O')
        return s

    if aao:
        name1 = [swe(n) for n in name1]
        name2 = [swe(n) for n in name2]

    import difflib

    score = np.zeros((len(name1),len(name2)))

    for n,n1 in enumerate(name1):

        score[n,:] = [difflib.SequenceMatcher(None, b, n1).ratio() for b in name2]

    return score


def splitnonalpha(s):
   pos = 1
   while pos < len(s) and s[pos].isalpha():
      pos+=1
   return (s[:pos], s[pos:])

def read_excel_table(file,worksheet = None,headers = [],srow = 1):
    """ Reads any excel data table into list. Each row is stored as a dictionary 
    in the list, with it's values stored using the column names as keys. Assumes
    the excel file contains headers in the first row. 
    Inputs: 
        file - complete file path 
        worksheet - name of worksheed to read 
        headers - list of headers, if different from headers specified in sheet
        srow - starting row 
    Outputs:
        data - list of dictionaries, one for each data row
        fields - dictionary keys, same as headers
    """
    
    wb = openpyxl.load_workbook(file)
    
    if worksheet is None: # select first worksheet
        ws = wb.worksheets[0]
    else: # select named worksheet
        ws = wb[worksheet]

    # first row contains header
    if headers == [] or headers.__len__() != ws.max_column:
        fields = [ws.cell(srow,i).value for i in range(1,ws.max_column+1)]
    else:
        fields = headers
            
    # Header may span multiple rows, increment row counter until we find
    # first non-empty cell
    srow = srow + 1
    while ws.cell(srow,1).value is None:
        srow = srow + 1
        
    # read all rows as dicts into list 
    data = []
    for i in range(srow,ws.max_row+1):
        d = {}
        for j in range(1,ws.max_column+1):
            d[fields[j-1]] = ws.cell(i,j).value
        data.append(d)    
        
    wb.close()
    
    return (data,fields)

def new_zero_dict(params):
    """ Create a new dictionary with zero/empty fields taken from params """
    d = {}
    for p in params:
        #d[p[0]] = p[1]()
        if p[1] is list:
            d[p[0]] = None
        else:
            d[p[0]] = p[1]()
    return d

def new_duplicate_dict(d):
    """ Create a duplicate of the given dictionary """
    d1 = {}
    for p in d.keys():
        d1[p] = d[p]
    return d1

def find_str(s,l):
    """ Find index of occurence of string s in list l """
    idx = 0
    while idx < l.__len__() and l[idx] != s:
        idx = idx + 1
    return idx

def format_comment(comment, max_line_length):
    """ Break up a string into several lines of given maximum length
    """
    #accumulated line length
    ACC_length = 0
    words = comment.split(" ")
    formatted_comment = ""
    for word in words:
        #if ACC_length + len(word) and a space is <= max_line_length 
        if ACC_length + (len(word) + 1) <= max_line_length:
            #append the word and a space
            formatted_comment = formatted_comment + word + " "
            #length = length + length of word + length of space
            ACC_length = ACC_length + len(word) + 1
        else:
            #append a line break, then the word and a space
            formatted_comment = formatted_comment + "\n" + word + " "
            #reset counter of length to the length of a word and a space
            ACC_length = len(word) + 1
    return formatted_comment

def intersection(lst1,lst2):
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def str_to_date(strdate,timestamp=True):
    """ Take a string with a date and return datetime object
    Allowed formats: 
        'YYYYMMDD'
        'YYYY-MM-DD'
        'YYYYMMDD:HH'
        'YYYY-MM-DD:HH'
        'YYYYMMDD:HHMM
    """
    year = int(strdate[0:4])
    if strdate[4] == '-':
        month = int(strdate[5:7])
        day = int(strdate[8:10])
        idx = 10
    else:
        month = int(strdate[4:6])
        day = int(strdate[6:8])
        idx = 8
    if strdate.__len__() > idx:
        hour = int(strdate[idx+1:idx+3])
        if strdate.__len__() - idx > 3:
            min = int(strdate[idx+3:idx+5])
        else:
            min = 0
        if hour == 24:
            hour = 0
            day += 1
        d = datetime.datetime(year,month,day,hour,min)
    else:
        d = datetime.datetime(year,month,day)
    if timestamp:
        return pd.Timestamp(d)
    else:
        return d
            

def week_to_date(weekstr):
    """ Given week in format 'YYYY:WW' return datetime object with date for start of week
    """
    year = int(weekstr[0:4])
    week = int(weekstr[5:7])
    return datetime.datetime(year,1,1) + datetime.timedelta(days=(week-1)*7)

def week_to_range(week,year):
    """ Return first and last hour in week in given year """
    
    start = datetime.datetime(year,1,1) + datetime.timedelta(days=7*(week-1))
    end = start + datetime.timedelta(days=7) + datetime.timedelta(seconds=-3600)
    if (end + datetime.timedelta(days=7)).year > year:
        end = datetime.datetime(year,12,31,23)
        
    return (start,end)

def date_to_week(date):
    """ Given datetime, find the week and return string with format 'YYYY:WW'
    Better version without looping
    """
    date0 = datetime.datetime(date.year,1,1)
    diff = date-date0
    nhours = diff.days*24 + diff.seconds / 3600
    widx = min(52,int(np.floor_divide(nhours,168)) + 1)
    return f'{date.year}:{weekdd[widx-1]}'


def increment_week(weekstr):
    """ Given week in format 'YYYY:WW', return next week """
    year = int(weekstr.split(':')[0])
    week = int(weekstr.split(':')[1])
    if week < 52:
        return '{0}:{1}'.format(year,weekdd[week])
    else:
        return '{0}:01'.format(year+1)
    
def decrement_week(weekstr):
    """ Given week in format 'YYYY:WW', return previous week """
    year = int(weekstr.split(':')[0])
    week = int(weekstr.split(':')[1])
    if week > 1:
        return '{0}:{1}'.format(year,weekdd[week-2])
    else:
        return '{0}:52'.format(year-1)    
    

def hours_in_week(weekstr):
    """ Number of hours in week """
    td = week_to_date(increment_week(weekstr))-week_to_date(weekstr)
    return td.days*hours_per_day + td.seconds/seconds_per_hour
    
def diff_hours(t1,t2):
    """ Number of hours between two dates """
    return (t2-t1).days*hours_per_day + (t2-t1).seconds/seconds_per_hour
    
def duration_curve(df,descending=True):
    """ Create duriation curve from time series """
    
    if type(df) is pd.DataFrame:
        if descending:
            return pd.DataFrame(np.sort(df.values,axis=0)[::-1],index=range(1,df.index.__len__()+1),columns=df.columns)
        else:
            return pd.DataFrame(np.sort(df.values,axis=0),index=range(1,df.index.__len__()+1),columns=df.columns)
    elif type(df) is pd.Series:
        if descending:
            return pd.Series(np.sort(df.values)[::-1],index=range(1,df.index.__len__()+1))
        else:
            return pd.Series(np.sort(df.values),index=range(1,df.index.__len__()+1))
    else:
        print("duration_curve: unknown data type")
        return None
    
def compact_xaxis_ticks(f,ax):
    
    # check if there are any minor tick labels, only then make labels more compact
    minor_ticks = False
    f.canvas.draw()
    for t in ax.xaxis.get_minor_ticks():
        if t.label.get_text() != '':
            minor_ticks = True
            break
    if minor_ticks:
        labls = []
        for t in ax.xaxis.get_major_ticks():
            
            tx = t.label.get_text()
            #print(repr(tx))
            # replace double linebreak by single linebreak
            tx = tx.replace('\n\n','\n')
            s = re.search('\n\d',tx) # search linebreak followed by digit
            if not s is None: # replace \n by ' '
                tx = tx[:s.span()[0]] + ' ' + tx[s.span()[0]+1:]
            labls.append(tx)
        ax.set_xticklabels(labls) 
    
def find_overlimit_events(s,thrs):
    """
    Given time series, find events when a certain threshold level is exceeded
    
    Used to compute information about curtailment events
    Returns:
        list of events, one tuple for each event:
            (start,end)
    """
    flag = False
    events = []
    for idx in s.index:
        if not flag and s.at[idx] > thrs:
            # start of curtailment event
            start = idx
            flag = True
        elif flag and (s.at[idx] < thrs or idx == s.index[-1]):
            # end of curtailment event
            end = idx
            events.append((start,end))
            flag = False
    return events
#def impute_values(df,zeros=True):
#    """ Impute missing (and zero if zeros) values in df """
#    
#    for col in df.columns:
#        if zeros:
#            # replace zeros with missing
#            nzero = (df[col] == 0).sum()
#            if nzero > 0:
#                df.loc[ df[col] == 0, col] = np.nan
#        nmiss = df[col].isna().sum()
#        if nmiss > 0:
#            df.loc[:,col] = df.loc[:,col].interpolate('linear')
#            if zeros:
#                print(f"Imputing {nmiss} (of which {nzero} zeros) values for {name}[{col}]")
#            else:
#                print('Imputing {0} values for {1}[''{2}'']'.format(nmiss,name,col))
  
def cet2utc(time):
    """ Convert CET time to UTC time (i.e. subtract one hour """
    return (str_to_date(time) + datetime.timedelta(hours=-1)).strftime('%Y%m%d:%H')

def utc2cet(time):
    """ Convert UTC time to CET = UTC + 1 time """
    return (str_to_date(time) + datetime.timedelta(hours=1)).strftime('%Y%m%d:%H')


def create_select_list(l):
    """ Create a string containing all elements in the list l:
        '("l[0]", "l[1]", ... , "l[-1]")'
        Used for conditional selects in Sqlite
    """
    s = '('
    for idx, cat in enumerate(l):
        if idx > 0:
            if type(cat) is str:
                s += ",'{0}'".format(cat)
            else:
                s += f",{cat}"
        else:
            if type(cat) is str:
                s += "'{0}'".format(cat)
            else:
                s += f"{cat}"

    s += ')'
    return s


def find_peaks(ts, mindist=100):
    """
    Find peaks in time series

    :param ts:
    :return:
    """

    extreme_value = -np.inf
    extreme_idx = 0
    peakvalues = []
    peaktimes = []
    find_peak = True
    idx = 0
    for r in ts.iteritems():
        # print(r)
        if find_peak:
            # look for maximum
            if r[1] > extreme_value:
                # update current maximum point
                extreme_value = r[1]
                extreme_idx = idx
            elif r[1] + mindist < extreme_value:
                # consider current maximum a peak
                peakvalues.append(extreme_value)
                peaktimes.append(extreme_idx)
                # update current maximum
                extreme_value = r[1]
                extreme_idx = idx
                find_peak = False
        else:
            # look for minimum
            if r[1] < extreme_value:
                # update value
                extreme_value = r[1]
                extreme_idx = idx
            elif r[1] - mindist > extreme_value:
                extreme_value = r[1]
                extreme_idx = idx
                find_peak = True
        idx += 1
    return peakvalues, peaktimes


def find_convex_hull(x, y):
    # find points corresponding to convex hull of increasing set of points x and y

    # ts = ts.copy(deep=True)
    # # convert to numeric time
    # ts.index = pd.to_timedelta(pd.Series(ts.index)).dt.total_seconds()

    hull_idxs = [0]
    idx = 0  # index for idxs

    while idx < x.__len__() - 1:
        # find slopes of all lines connecting x[idx] with x[idx+i], i > 0
        slopes = [(y[i] - y[idx]) / (x[i] - x[idx]) for i in range(idx + 1, x.__len__())]
        imax = np.argmax(slopes)
        # print(imax)
        # append new point to x
        hull_idxs.append(imax + 1 + idx)
        # update idx
        idx = idx + imax + 1

    hull_idxs.append(x.__len__() - 1)
    return hull_idxs

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_abs_axis_pos(rel_pos, ax):
    """
    Find coordinates in terms of axis quantities, given relative coordinates from 0 to 1,
    so that e.g. (1,1) refers to (xmax,ymax)

    :param rel_pos: relative coordinates
    :param ax: axis
    :return: absolute coordinates
    """

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    return (xlims[0] + rel_pos[0] * (xlims[1] - xlims[0]), ylims[0] + rel_pos[1] * (ylims[1] - ylims[0]))


def hour_min_sec(seconds):
    hour = np.floor_divide(seconds,3600)
    min = np.floor_divide(seconds-hour*3600,60)
    sec = seconds - hour*3600 - min*60
    return (hour,min,sec)

def interp_time(dates,df):
    """
    Interpolate instantaneous values from given dataframe (assuming linear change in time)

    df - dataframe with data, must cover whole range in dates
    dates - time stamps at which interpolated data is desired
    """
    # create dataframe with all time values
    df_idx_unique = [i for i in df.index if i not in dates]
    time_index = df_idx_unique + [i for i in dates]
    time_index.sort()
    df2 = pd.DataFrame(dtype=float,index=time_index,columns=df.columns)
    # put original data in dataframe
    df2.loc[df.index,df.columns] = df
    # interpolate assuming linear change in time
    df2.interpolate(method='time',inplace=True)
    # drop original values
    df2.drop(df_idx_unique,axis=0,inplace=True)
    return df2

def week2date_iso(week1,week2=None):
    """
    Return first and last hour in week in the given week1 with format 'YYYY:WW'
    If week2 is specified this is the ending week
    :param weekstr:
    :return:
    """
    wfmt = '%G:%V-%u'
    wstart = datetime.datetime.strptime(f'{week1}-1',wfmt)
    if week2 is None:
        wend = datetime.datetime.strptime(f'{week1}-7',wfmt) + datetime.timedelta(hours=23)
    else:
        wend = datetime.datetime.strptime(f'{week2}-7',wfmt) + datetime.timedelta(hours=23)
    return wstart,wend

def week2str(year,week):
    return f'{year}:' + f'{week}'.zfill(2)

def date2week_iso(date):
    """
    Return iso week 'YYYY:WW' of given date (hour)
    Accepted date formats:
    - datetime
    - string:
    'YYYYMMDD'
    'YYYYMMDD:HH'
    """
    if type(date) is str:
        isodate = datetime.datetime.strptime(date[:8],'%Y%m%d').isocalendar()
    else:
        isodate = date.isocalendar()
    return f'{isodate[0]}:' + f'{isodate[1]}'.zfill(2)

def interpolate_weekly_values(df,method='linear'):
    """
    Given dataframe df with weekly (energy values), create hourly dataframe either with constant values
    or a linear interpolation that preserves the energy content of the given data
    :param method - constant/linear to specify method for interpolation
    :return:
    """

    timerange = pd.date_range(start=df.index[0],end=df.index[-1],freq='H')
    df_hour = pd.DataFrame(dtype=float,columns=df.columns,index=timerange)

    if method == 'linear':
        t2 = df.index[0]
        for i,t in enumerate(df.index): # fix values at shifts between weeks
            df_hour.loc[t,:] = 0.5*(df.loc[t,:]+df.loc[t2,:]) / 168
            t2 = t # save index of last week

        # fix values at middle of week, to get correct weekly inflow
        for i,t in enumerate(df.index[:df.shape[0]-1]):
            avg = df.loc[t,:] / 168 # average inflow should be preserved
            y1 = df_hour.loc[t,:]
            y2 = df_hour.loc[t+datetime.timedelta(days=7),:]
            tmid = t + datetime.timedelta(hours=84) # middle of week
            df_hour.at[tmid,:] = 2*avg - 0.5*y1 - 0.5*y2 # set middle value to preserve weekly inflow
        # linear interpolation
        df_hour.interpolate(method='linear',inplace=True)

    else:
        # simple filling
        df_hour.loc[df.index,:] = df.loc[:,:] / 168
        df_hour.ffill(inplace=True)

    return df_hour

def time_to_bin(t,binstart,binsize):
    """
    Find the index of the bin for given time t
    Used to map time value to correct cost coefficient when using shifted costs
    If binstart = XXXX0101:00 and binsize=168, this will return the index of the week
    Note that the year of t is not considered, i.e. it is assumed that the time stamp is in the same
    year as the bins
    :param binstart - date when bins start being counted (i.e. first hour of first bin)
    :param binsize - number of hours in each bin
    :return: ibin - index of bin
    """
    # update binstart to same year as t (assumes binstart doesnt occur on leap day)
    bs = datetime.datetime(t.year,binstart.month,binstart.day,binstart.hour)
    if bs > t:
        # decrease by one year
        bs = datetime.datetime(t.year-1,binstart.month,binstart.day,binstart.hour)
    diff = t - bs
    # count hours between binstart and given value
    hdiff = diff.days*24 + diff.seconds/3600
    return int(np.floor_divide(hdiff,binsize))

def err_func(p,phat):
    """ Compute mean absolute error from time series """
    mae = np.mean(np.abs(p-phat))
    norm = 0.5*(np.mean(np.abs(p))+np.mean(np.abs(phat)))
    return mae,mae/norm,norm


def curtailment_statistics(curtail,potential,curstat_cols = ['GWh','%','#','avg len','avg GW','avg %']):
    """
    Given time series for wind potential and wind curtailment, compute curtailment
    statistics

    Returns:
        Series(['%','GWh','#','avg len','avg %','avg GW'])
        % - total curtailment as percent of available generation
        GWh - total curtailment in GWh
        # - number of events
        avg len - average length of curtailment events
        avg % - share of curtailment as percent of available production
                during hours with curtailment
        avg GW - avergae curtailment in GW during hours with curtailment
    """
    thrs = 1e-3
    # find curtailment events
    events = find_overlimit_events(curtail,thrs)

    s = pd.Series(dtype=float,index=curstat_cols)

    s.at['GWh'] = curtail.sum()
    GWh_max = potential.sum()
    if GWh_max > 0:
        s.at['%'] = 1e2 * s.at['GWh'] / GWh_max
    else:
        s.at['%'] = 0
    s.at['#'] = events.__len__()

    if s.at['#'] > 0:
        # number of hours with curtailment
        nhours = sum(int(24*(e[1]-e[0]).days+(e[1]-e[0]).seconds/3600) for e in events)
        # curtailment during hours with curtailment (i.e. basically same as GWh)
        curtot = sum(curtail.loc[ pd.date_range(e[0],e[1],freq='H') ].sum() for e in events)
        # available production during hours with curtailment
        curpot = sum(potential.loc[ pd.date_range(e[0],e[1],freq='H') ].sum() for e in events)

        s.at['avg len'] = nhours / s.at['#']
        s.at['avg GW'] = curtot / nhours
        s.at['avg %'] = 1e2 * curtot / curpot

    else:
        s.at['avg len'] = 0
        s.at['avg GW'] = 0
        s.at['avg %'] = 0

    return s

if __name__ == '__main__':

    pass
    # week_to_range(1,2018)
    #
    # dates = week_to_range(1,2018)
    #
    # r = diff_hours(dates[0],dates[1])
    #
    # n = hours_in_week('2018:52')


