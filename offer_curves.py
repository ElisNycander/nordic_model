# -*- coding: utf-8 -*-
"""
Class for storing supply curves and calculating marginal costs

Created on Thu Feb  7 15:34:33 2019

@author: elisn
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SupplyCurve():
    """ Has panda dataframe with list  of bids
    
    One or many generators may be added to the supply curve. The generators must be in the form
    of a panda data frame, with the columns ['c2','c1','pmax']
    The marginal cost of a generator is given by MC = 2*c2*q + c1, where q ranges from 0 to pmax
    Hence a generator with c2 = 0 has constant marginal cost
    (Thus note that the coefficients are those for the Total Cost function)
    
    It is also possible to add bids, which then must have the columns [cap,mc_min,mc_max]
    
    Note that the internal functions use the bid structure. 
    
    Class methods:
        price2quantity(price) - calculates the quantity offered for a given price, straightforward calculation
        quantity2price(quantity) - calculates price required for given quantity
                                    not straightforward since constant bids produce discrete jumps in the offered quantity
        plot() - plots supply curve
    
    """
    def __init__(self,bids = pd.DataFrame(columns=['cap','mc_min','mc_max']),gens = pd.DataFrame(columns=['c2','c1','pmax']) ):
        

        self.bids = bids.append(get_generator_bids(gens),ignore_index=True).sort_values(by=['mc_min','mc_max'])
        self._calculate_inflection_points_()


    def add_bids(self,bids):
        """ Add bids to supply curve, in the form of a data frame """
        self.bids = self.bids.append(bids,ignore_index=True).sort_values(by=['mc_min','mc_max'])
        self._calculate_inflection_points_()
        
    def add_gens(self,gens):
        """ Add generators with c1, c2, pmax coefficients to supply curve """
        self.bids = self.bids.append(get_generator_bids(gens),ignore_index=True).sort_values(by=['mc_min','mc_max'])
        self._calculate_inflection_points_()
        
    def price2quantity(self,price):
        """ Calculate the offered quantity for a given price """

        # loop over bids, calculate offer by each
        quantity = 0
        for i in self.bids.index:
            if price >= self.bids.loc[i,'mc_min']:
                if self.bids.loc[i,'mc_min'] != self.bids.loc[i,'mc_max']: # variable MC
                    q = (price - self.bids.loc[i,'mc_min'])/(self.bids.loc[i,'mc_max']-self.bids.loc[i,'mc_min'])*self.bids.loc[i,'cap']
                    if q > self.bids.loc[i,'cap']:
                        q = self.bids.loc[i,'cap']
                    quantity += q
                else: # fixed MC 
                    quantity += self.bids.loc[i,'cap']
            else:
                # mc_min exceeds price, can exit as bids are sorted by increasing mc_min
                return quantity
        return quantity

    def _calculate_inflection_points_(self):
        """ Find all inflection points in the supply curve """
   
        ppoints = []
        for i in self.bids.index:
            if self.bids.loc[i,'mc_min'] not in ppoints:
                ppoints.append(self.bids.loc[i,'mc_min'])
            if self.bids.loc[i,'mc_max'] not in ppoints:
                ppoints.append(self.bids.loc[i,'mc_max'])
        ppoints.sort()
        
        # find curresponding quantities
        qpoints = []
        for point in ppoints:
            qpoints.append(self.price2quantity(point))  
        
        self.xprice = ppoints
        self.xquant = qpoints

    def quantity2price(self,quantity,plot=False,verbose=False):
        """ Calculate minimum price needed for given quantity """
        
        idx = 0
        while True:
            if idx == self.xprice.__len__():
                    # quantity > qmax, not enough capacity
                    if verbose:
                        print("Insufficient capacity: {0} MW available, but quantity = {1:.3}".format(self.xquant[-1],quantity))
                    #return np.nan
                    p = np.nan
                    break
            elif self.xquant[idx] < quantity:
                idx += 1 # go to next price level
            else:
                if idx == 0:
                    # quantity <= 0 - return lowest marginal cost
                    #print("Non-positive quantity = {0:.3}, returning lowest available MC".format(quantity))
                    #return self.xprice[0]
                    p = self.xprice[0]
                    break
                elif self.xquant[idx] == quantity:
                    # price corresponds exactly to quantity
                    #return self.xprice[idx]
                    p = self.xprice[idx]
                    break
                else:
                    # check if offer curve is linear by evaluating quantity between prices
                    if self.price2quantity(self.xprice[idx-1]+(self.xprice[idx]-self.xprice[idx-1])/2) > self.xquant[idx-1]:
                        # if offer curve is linear, interpolate to find correct price
                        # Note: Cannot interpolate linearly to next intersection point, as there
                        # the curve may consist of a linear horizontal section to the next point
                        # Thus we must instead find the inverse slope by summing the inverse slopes
                        # of linear bids at this point
                        # use inverse slope at price xprice[idx] for interpolation
                        p = self.xprice[idx-1] + (quantity-self.xquant[idx-1]) / self._find_slope_(self.xprice[idx])
                        if p > self.xprice[idx]: # cap price increase up to xprice[idx]
#                            if idx == 3:
#                                print(p)
#                                pass
                            p = self.xprice[idx]
                        #return p
                        break
                    else:
                        # else return this price
                        p = self.xprice[idx]
                        #return self.xprice[idx]
                        break
        
        if plot:
            # plot supply curve with determined point
            self.plot(qpoints=[quantity],ppoints=[p])
        return p
                    
    def _find_slope_(self,price):
        """ Find the slope of the supply curve, in MW/EUR (quantity/price) for given price """
        
        # loop over all linear bids and see which are active in this price range
        slope = 0 # slope in MW/EUR
        for index in self.bids.index:
            if self.bids.loc[index,'mc_min'] != self.bids.loc[index,'mc_max'] and \
                price > self.bids.loc[index,'mc_min'] and price <= self.bids.loc[index,'mc_max']:
                slope += self.bids.loc[index,'cap']/(self.bids.loc[index,'mc_max']-self.bids.loc[index,'mc_min'])
        return slope
                    
    def plot(self,qpoints=[],ppoints=[]):
        """ Plot supply curve """
        
        x_quantity = np.linspace(0,self.xquant[-1])
        y_price = np.array([self.quantity2price(x) for x in x_quantity])
        
        y2_price = np.linspace(self.xprice[0],self.xprice[-1])
        x2_quantity = np.array([self.price2quantity(p) for p in y2_price])
        
#        # merge data points into single array
#        x = np.array([x for x,_ in sorted(zip(list(x_quantity)+list(x2_quantity),list(y_price)+list(y2_price)))])
#        y = np.array([y for _,y in sorted(zip(list(x_quantity)+list(x2_quantity),list(y_price)+list(y2_price)))])
#        
        plt.plot()
        plt.plot(x_quantity,y_price,'*')
        plt.plot(x2_quantity,y2_price,'*')
        #plt.plot(x,y)
        # add given points to plot
        if qpoints.__len__() > 0:
            plt.plot(np.array(qpoints),np.array(ppoints),'r*')
        plt.grid()
        plt.xlabel('MW')
        plt.ylabel('EUR/MWh')
        plt.title('Supply curve')
        plt.legend(['quantity2price','price2quantity'])
        plt.show()
        
    def get_curve(self):
        """ Return x and y vector with points to plot the offer curve """
        
        x_quantity = np.linspace(0,self.xquant[-1])
        y_price = np.array([self.quantity2price(x) for x in x_quantity])
        
        y2_price = np.linspace(self.xprice[0],self.xprice[-1])
        x2_quantity = np.array([self.price2quantity(p) for p in y2_price])
        
        # merge data points into single array
        x = np.array([x for x,_ in sorted(zip(list(x_quantity)+list(x2_quantity),list(y_price)+list(y2_price)))])
        y = np.array([y for _,y in sorted(zip(list(x_quantity)+list(x2_quantity),list(y_price)+list(y2_price)))])
        
        return x,y
        
        
def get_generator_bids(gens):
    """ Takes a panda dataframe with generator info, and returns a dataframe with bids
    with the columns [cap,mc_min,mc_max]
    cap - total capacity of bid
    mc_min - minimum marginal cost (=c1)
    mc_max - maximum marginal cost (=2*c2)
    """
    bids = pd.DataFrame(columns=['cap','mc_min','mc_max'])
    bids.loc[:,'cap'] = gens.loc[:,'pmax']
    bids.loc[:,'mc_min'] = gens.loc[:,'c1']
    bids.loc[:,'mc_max'] =  gens.loc[:,'pmax'] * gens.loc[:,'c2']*2 + gens.loc[:,'c1']
    bids.index = list(range(bids.__len__()))
    return bids


if __name__ == "__main__":
    
    with open('Data/generators.pkl','rb') as f:
        gens = pickle.load(f)
     
    
#    gens = pd.DataFrame(columns=['c1','c2','pmax'],index=[1,2])
#    gens.loc[1,:] = [10,0,10000]
#    gens.loc[2,:] = [20,0,10000]
#    gens.loc[3,:] = [15,0.0005,10000]

 
    s = SupplyCurve(gens=gens)
    s.plot()
    
    s.add_bids(pd.DataFrame(np.array([[10000,10,10],[10000,80,80]]),columns=['cap','mc_min','mc_max']))

    s.plot()
    
    x,y = s.get_curve()
    plt.plot(x,y)
    
    