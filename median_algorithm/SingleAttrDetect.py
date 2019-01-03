# SingleAttrDetect
from datetime import datetime
from urllib.parse import parse_qs
from urllib.parse import urlparse

import numpy as np
from MesoPy import Meso
from pandas.io.json import json_normalize
import pandas as pd
from statsmodels import robust
from scipy.stats import norm

import requests
import json


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


class MesokNN:
    _timeformat = '%s'

    def __init__(self, query, stid, k=5):
        if isinstance(query, str):
            query_args = parse_qs(query)
        elif isinstance(query, dict):
            query_args = query

        if query_args['pmode'][0] != 'totals':
            raise ValueError('pmode must be totals')

        self._ctr_station = stid                # center station
        self._token = query_args['token'][0]    # api token
        self._start = query_args['start'][0]    # start date UTC
        self._end = query_args['end'][0]        # end date UTC
        self._units = query_args['units'][0]    # units used

        self.meso_instance = Meso(token=self._token)  # MesoPy instance for api requests.

        self.neighborhood = self._neighbors(k=k)

    def _neighbors(self, k):
        '''
        Returns the k nearest stations and their observations as a pandas dataframe.
        :param validate: validate record period or not.
        :param k: number of neighbors.
        :return: dataframe of k neighbors including some metadata and observations.
        '''

        # compute the max timedelta of observations
        start = datetime.strptime(self._start, '%Y%m%d%H%M')
        end = datetime.strptime(self._end, '%Y%m%d%H%M')
        delta = end.timestamp() - start.timestamp()

        _radius_arg = self._ctr_station + ',50'
        _accum_label = 'ACCUM_' + str(int(delta / 86400)).strip() + '_DAYS'

        request_df = json_normalize(self.meso_instance.precip(self._start, self._end, pmode='totals',
                                                              units=self._units, timeformat=self._timeformat,
                                                              radius=_radius_arg, limit=str(k))['STATION'])

        precip_col = np.full(request_df.shape[0], np.nan, dtype='float64')
        epoch_delta_col = np.full(request_df.shape[0], 0, dtype='int')

        for i, row in request_df.iterrows():  # extract precip observation into it's own column
            if len(row['OBSERVATIONS.precipitation']) > 0:
                _dict = row['OBSERVATIONS.precipitation'][0]
                epoch_delta_col[i] = (int(_dict['last_report']) - int(_dict['first_report']))
                precip_col[i] = _dict['total']

        request_df[_accum_label] = precip_col            # precip data column
        request_df['EPOCH_TIMEDELTA'] = epoch_delta_col  # used to filter out erroneous time periods

        request_df = request_df[abs(request_df['EPOCH_TIMEDELTA'] - delta) < .1 * delta]

        self.values = request_df[_accum_label].values

        return request_df[['STID', 'LATITUDE', 'LONGITUDE',  _accum_label]].set_index('STID', drop=True)

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

    def fit(self, data, k=5, attr_func=lambda x: x, comp_func=lambda f, g: f-g,  spatial_ind=None, stid_ind='index',
            attr_ind=0, query_string=None):
        # data checks

        # class attribute assignment
        self.data = data
        self.spatial_ind = spatial_ind
        self.attr_func = attr_func
        self.comp_func = comp_func
        self.stid_ind = stid_ind

        if spatial_ind:  # classical kNN
            raise NotImplemented
        else:
            if self.stid_ind == 'index':
                stids = data.index.values
            else:
                stids = data.iloc[:, self.stid_ind].values

            self.neighbor_values = np.empty(stids.shape[0])
            self.attr_values = np.empty(stids.shape[0])
            self.comp_values = np.empty(stids.shape[0])

            # compute neighborhoods, attribute function, neighborhood function, and comp function for each data point
            for i, stid in enumerate(stids):
                knn_i = MesokNN(parse_qs(query_string), stid, k=k).values     # neighborhood observations
                self.neighbor_values[i] = np.median(knn_i)                    # neighborhood function
                self.attr_values[i] = self.attr_func(data.iloc[:, attr_ind][i])  # attribute function
                self.comp_values[i] = self.comp_func(self.attr_values[i], self.neighbor_values[i])  # comp function

    def detect(self, alpha=0.01):


        mu_star = np.median(self.comp_values)
        sig_star = robust.mad(self.comp_values)

        ys = abs((self.comp_values-mu_star)/sig_star)

        print(self.neighbor_values)
        print(self.attr_values)
        print(self.comp_values)
        print(ys)


if __name__ == '__main__':
    url = 'http://api.synopticlabs.org/v2/stations/precipitation?token=demotoken&radius=wbb,2&pmode=totals&start=201803010000&end=201810140000&units=english&timeformat=%s'

    m = Meso(token='demotoken')
    df = json_normalize(m.precip(start='201803010000', end='201810140000', pmode='totals', radius='ksltc,5', timeformat='%s', units='english')['STATION'])

    precip_col = np.full(df.shape[0], np.nan, dtype='float64')
    epoch_delta_col = np.full(df.shape[0], 0, dtype='int')

    start = datetime.strptime('201803010000', '%Y%m%d%H%M')
    end = datetime.strptime('201810140000', '%Y%m%d%H%M')
    delta = end.timestamp() - start.timestamp()

    _accum_label = 'ACCUM_' + str(int(delta / 86400)).strip() + '_DAYS'

    for i, row in df.iterrows():  # extract precip observation into it's own column
        if len(row['OBSERVATIONS.precipitation']) > 0:
            _dict = row['OBSERVATIONS.precipitation'][0]
            epoch_delta_col[i] = (int(_dict['last_report']) - int(_dict['first_report']))
            precip_col[i] = _dict['total']

    df[_accum_label] = precip_col  # precip data column
    df['EPOCH_TIMEDELTA'] = epoch_delta_col  # used to filter out erroneous time periods

    df = df[abs(df['EPOCH_TIMEDELTA'] - delta) < .01 * delta]
    df = df[['STID', _accum_label]].set_index('STID', drop=True)

    print(df)

    d = SingleAttrDetect()
    d.fit(df, query_string='&units=english&token=demotoken&pmode=totals&start=201803010000&end=201810140000&stid=wbb', k=10)

    d.detect()