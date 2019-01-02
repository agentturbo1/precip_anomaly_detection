# SingleAttrDetect
from datetime import datetime
from urllib.parse import parse_qs
from urllib.parse import urlparse

import numpy as np
from MesoPy import Meso
from pandas.io.json import json_normalize

'''
MesokNN Class
Represents a neighborhood of size k that consists of the k nearest neighbors, and their observations of a given station
with the same Meso API request. Precipitation only.
'''


class MesokNN:
    timeformat = '%s'

    def __init__(self, api_request, stid):
        api_url = urlparse(api_request)

        if api_url.path != '/v2/stations/precipitation':
            pass  # TODO add exception

        query_args = parse_qs(api_url.query)

        self._ctr_station = stid                # center station
        self._token = query_args['token'][0]    # api token
        self._start = query_args['start'][0]    # start date UTC
        self._end = query_args['end'][0]        # end date UTC
        self._units = query_args['units'][0]    # units used

        self.meso_instance = Meso(token=self._token)  # MesoPy instance for api requests.

    def neighbors(self, k=5):
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

        _radius_arg = self._ctr_station + ',30'
        _accum_label = 'ACCUM_' + str(int(delta / 86400)).strip() + '_DAYS'

        inc = 0
        while True:
            request_df = json_normalize(self.meso_instance.precip(self._start, self._end, pmode='totals',
                                                                  units=self._units, timeformat=self.timeformat,
                                                                  radius=_radius_arg, limit=str(k + inc))['STATION'])
            inc += 1

            precip_col = np.full(request_df.shape[0], np.nan, dtype='float64')
            epoch_delta_col = np.full(request_df.shape[0], 0, dtype='int')

            for i, row in request_df.iterrows():  # extract precip observation into it's own column
                if len(row['OBSERVATIONS.precipitation']) > 0:
                    _dict = row['OBSERVATIONS.precipitation'][0]
                    epoch_delta_col[i] = (int(_dict['last_report']) - int(_dict['first_report']))
                    precip_col[i] = _dict['total']

            request_df[_accum_label] = precip_col
            request_df['EPOCH_TIMEDELTA'] = epoch_delta_col  # used to filter out erroneous time periods

            request_df = request_df[abs(request_df['EPOCH_TIMEDELTA'] - delta) < .1 * delta]

            if request_df.shape[0] == k or k+inc > 50:
                break

        return request_df[['STID', 'LATITUDE', 'LONGITUDE',  _accum_label]].set_index('STID', drop=True)



if __name__ == '__main__':
    url = 'http://api.synopticlabs.org/v2/stations/precipitation?token=demotoken&radius=wbb,15&pmode=totals&start=201803010000&end=201810140000&units=english'
    knn = MesokNN(url, 'DPG31')
    neighborhood = knn.neighbors(10)