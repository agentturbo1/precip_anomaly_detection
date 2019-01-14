# SingleAttrDetect
from datetime import datetime

import numpy as np
from MesoPy import Meso
from pandas.io.json import json_normalize
from statsmodels import robust
from MesoTools.MesoDataframes import precip_dataframe
from scipy.stats import norm


class NotFitError(Exception):
    pass

'''
MesokNN Class
Represents a neighborhood of size k that consists of the k nearest neighbors, and their observations of a given station
with the same Meso API request. Precipitation only.

MesokNN represents a neighborhood of stations within the Synoptic API. Given a station and a set of observation 
parameters this object can find the k nearest neighbors with the same observation. e.g. 5 nearest neighbors of STID: WBB
for precipitation with start=20181225000, end=201812260000, pmode=totals, etc. 
'''


class PrecipMesokNN:
    _timeformat = '%s'

    def __init__(self, m, start, end, stid, k=5, **kwargs):
        if 'pmode' in kwargs and kwargs['pmode'] != 'totals':
            raise ValueError('pmode must be totals')

        self._ctr_station = stid  # center station
        kwargs['timeformat'] = self._timeformat
        kwargs['radius'] = self._ctr_station + ',50'
        kwargs['limit'] = str(k)

        self.meso_instance = m  # MesoPy instance for api requests.

        self.neighborhood = precip_dataframe(self.meso_instance, start, end, **kwargs)


'''
SingleAttrDetect Class
A model using a median algorithm that estimates a probability with given a comparison function for each data point.

This model, by default, relies on the Synoptic API to find the nearest neighbors instead of manually calculating knn 
using certain spatial attributes. If spatial column indices are passed then classical knn is calculated.
'''


class SingleAttrDetect:
    def __init__(self):
        self.data = None
        self.alpha = None
        self.spatial_ind = None
        self.attr_func = None
        self.comp_func = None
        self.stid_ind = None
        self.neighbor_values = None
        self.attr_values = None
        self.comp_values = None

    def fit(self, m, k=5, attr_func=lambda x: x, comp_func=lambda f, g: f-g,  spatial_ind=None, stid_ind='index',
            attr_ind=0, **kwargs):
        # temporary
        start = kwargs['start']
        end = kwargs['end']

        del kwargs['start']
        del kwargs['end']

        # class attribute assignment
        self.data = precip_dataframe(m, start, end, **kwargs)
        if self.data.empty or self.data.shape[0] < 5:
            raise ValueError('Not enough data for that query.')
        self.spatial_ind = spatial_ind
        self.attr_func = attr_func
        self.comp_func = comp_func
        self.stid_ind = stid_ind

        if spatial_ind:  # classical kNN
            raise NotImplemented
        else:
            if self.stid_ind == 'index':
                stids = self.data.index.values
            else:
                stids = self.data.iloc[:, self.stid_ind].values

            self.neighbor_values = np.empty(stids.shape[0])
            self.attr_values = np.empty(stids.shape[0])
            self.comp_values = np.empty(stids.shape[0])

            # compute neighborhoods, attribute function, neighborhood function, and comp function for each data point
            for i, stid in enumerate(stids):
                knn_i = PrecipMesokNN(m, start, end, stid, k=k, **kwargs).neighborhood.values  # neighborhood observations
                self.neighbor_values[i] = np.median(knn_i)                    # neighborhood function
                self.attr_values[i] = self.attr_func(self.data.iloc[:, attr_ind][i])  # attribute function
                self.comp_values[i] = self.comp_func(self.attr_values[i], self.neighbor_values[i])  # comp function

    def detect(self, alpha=0.01):
        mu_star = np.median(self.comp_values)
        sig_star = robust.mad(self.comp_values)

        ys = abs((self.comp_values-mu_star)/sig_star)
        outliers = ys >= abs(norm.ppf(alpha/2))

        return ys, outliers


if __name__ == '__main__':
    m = Meso(token='demotoken')
    df = precip_dataframe(m, '201803010000', '201810140000', pmode='totals', radius='dpg03,20', timeformat='%s')

    print(df)

    d = SingleAttrDetect()
    d.fit(m, k=8, start='201803010000', end='201810140000', pmode='totals', radius='dpg03,20', timeformat='%s')

    print(df[d.detect()[1]])
