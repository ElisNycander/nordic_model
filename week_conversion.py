
"""
Class to convert dates to weeks and vice versa

Week to date conversion:
ws - (1-7) the week day which is the "start" of the week, which means that week 1 for a given year is counted
as the first week containing the given week day
pw - True/False, count proper weeks, i.e. week starts with monday. If this is true, ws determines the year to
which the week belongs, but the week actually starts on the previous monday

Note: ws=4, pw=True corresponds to the standard ISO week format, in which the week starts on a Monday, but the year
to which the week belongs is the year of the corresponding
"""

import datetime
import pandas as pd

class WeekDef:

    def __init__(self,week_start=4,proper_week=True):
        self.ws = week_start
        self.pw = proper_week


    def week1(self,year):
        """ Return start of first week in year """
        t = datetime.datetime(year,1,1)
        while t.isocalendar()[2] != self.ws:
            t += datetime.timedelta(days=1)
        if self.pw:
            t += datetime.timedelta(days=1-self.ws)
        return t

    def week2date(self,week=1,year=2016):
        """ Return start of week """
        if type(week) is str:
            # asssume format 'YYYY:WW'
            year = int(week[:4])
            week = int(week[5:])
        t = self.week1(year)
        return t + datetime.timedelta(days=(week-1)*7)

    def date2week(self,t,sout=False):

        if type(t) is str:
            if t.__len__() <= 8:
                fmt = '%Y%m%d'
            elif t.__len__() <= 11:
                fmt = '%Y%m%d:%H'
            else:
                fmt = '%Y%m%d:%H%M'
            t = datetime.datetime.strptime(t,fmt)

        # pw = False: can find year by decreasing day until we reach ws
        # pw = True: day < ws: increase day until ws, day > ws: decrease day until ws
        # find previous starting week day
        tstart = t
        if not self.pw:
            while tstart.isocalendar()[2] != self.ws:
                tstart += datetime.timedelta(days=-1)
        else:
            while tstart.isocalendar()[2] < self.ws:
                tstart += datetime.timedelta(days=1)
            while tstart.isocalendar()[2] > self.ws:
                tstart += datetime.timedelta(days=-1)
        wyear = tstart.year
        # find number of this week
        wnum = 0
        # decrement by 1 week until we reach previous year
        while tstart.year == wyear:
            wnum += 1
            tstart += datetime.timedelta(days=-7)
        if sout: # return string YYYY:WW
            return f'{wyear}:' + f'{wnum}'.zfill(2)
        else:
            return wyear,wnum

    def range2weeks(self,t1='20180101:00',t2='20180301:00',sout=True):
        """ Return all weeks in given time range

        sout: True/False, if True, return weeks of format 'YYYY:WW', else return pandas date_range with starting hour
        of all weeks in range, and starting hour of first week outside of range
        """

        if type(t1) is str:
            if t1.__len__() != t2.__len__():
                raise ValueError('t1 and t2 must have same length!')
            if t2.__len__() <= 7:
                # format YYYY:WW
                tstart = self.week2date(t1)
                tend = self.week2date(t2)
            else:
                if t2.__len__() <= 8:
                    fmt = '%Y%m%d'
                elif t2.__len__() <= 11:
                    fmt = '%Y%m%d:%H'
                else:
                    fmt = '%Y%m%d:%H%M'

                tstart = datetime.datetime.strptime(t1,fmt)
                tend = datetime.datetime.strptime(t2,fmt)
        else: # t1,t2 datetime objects
            tstart = t1
            tend = t2

        if sout:
            weeks = [self.date2week(tstart,sout=True)]
            t = self.week2date(weeks[0]) + datetime.timedelta(days=7)
            # keep adding weeks until t2 is covered
            while t <= tend:
                weeks.append(self.date2week(t,sout=True))
                t += datetime.timedelta(days=7)
        else:
            dend = self.week2date(self.date2week(tend,sout=True))
            dstart = self.week2date(self.date2week(tstart,sout=True))
            weeks = pd.date_range(start=dstart,end=dend + datetime.timedelta(days=7),freq='7D')
            if weeks[-2] != dend:
                raise ValueError("Error, period not multiple of 7 days!")

        return weeks

    def long_years(self,startyear=2000,endyear=2016):

        w = self.range2weeks(f'{startyear}0101:00',f'{endyear+1}0101:00')
        return [y for y in range(startyear,endyear+1) if f'{y}:53' in w]



def verify_week_conversions():
    " Verify that result is the same as for functions based on iso calendar "

    from help_functions import str_to_date, week2date_iso, date2week_iso
    import pandas as pd

    dates = pd.date_range(start=str_to_date('20000101:00'),end=str_to_date('20201230'),freq='10H')

    # compare date2week and date2week_iso
    ws,pw = 4,True
    w = WeekDef(ws,pw)

    weeks = pd.DataFrame(index=dates,columns=['v2','iso'])
    for d in dates:
        weeks.at[d,'v2'] = w.date2week(d,sout=True)
        weeks.at[d,'iso'] = date2week_iso(d)

    #%% return starting hour of each week
    dates2 = pd.DataFrame(index=dates,columns=['v2','iso'])
    for d in dates:
        dates2.at[d,'v2'] = w.week2date(weeks.at[d,'v2'])
        dates2.at[d,'iso'] = week2date_iso(weeks.at[d,'iso'])[0]

    print(f'weeks: {(weeks["v2"]!=weeks["iso"]).sum()}')
    print(f'dates: {(dates2["v2"]!=dates2["iso"]).sum()}')

if __name__ == "__main__":
    pass

    # verify_week_conversions()

    w = WeekDef()
    # weeks = w.range2weeks('2008:01','2020:01')
    # years = [int(w[:4]) for w in weeks if ':53' in w]

    d = w.range2weeks('2021:01','2021:02',sout=False)


