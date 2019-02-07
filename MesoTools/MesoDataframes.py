from pandas.io.json import json_normalize
from datetime import datetime
import numpy as np
from MesoPy import Meso
import pandas


def precip_dataframe(m: Meso, start: str, end: str, **kwargs) -> pandas.DataFrame:
    '''
    Returns a pandas dataframe with STID as the index and the precip observation as the first column.
    :param m: Meso instance for retrieving data.
    :param start: start of observation window.
    :param end: end of observation window.
    :param kwargs: other key word arguments for the API.
    :return: the stations with a observation window within 0.05% of the expected window.

    The observation window is defined as (end - start). When creating the dataframe we filter out any observations that
    do not have that large of a window as they are un representative of the situation. The filter uses last_report and
    first_report from the API dictionary to determine if it is valid. (Returning this invalid data may be implemented)
    '''
    kwargs['timeformat'] = '%s'  # force timeformat
    df = json_normalize(m.precip(start=start, end=end, **kwargs)['STATION'])

    precip_col = np.full(df.shape[0], np.nan, dtype='float64')
    epoch_delta_col = np.full(df.shape[0], 0, dtype='int')

    start = datetime.strptime(start, '%Y%m%d%H%M')
    end = datetime.strptime(end, '%Y%m%d%H%M')
    delta = end.timestamp() - start.timestamp()

    if 'units' in kwargs:
        _accum_label = 'ACCUM_' + str(int(delta / 86400)).strip() + '_DAYS[' + kwargs['units'] + ']'
    else:
        _accum_label = 'ACCUM_' + str(int(delta / 86400)).strip() + '_DAYS[mm]'

    for i, row in df.iterrows():  # extract precip observation into it's own column
        if len(row['OBSERVATIONS.precipitation']) > 0:
            _dict = row['OBSERVATIONS.precipitation'][0]
            epoch_delta_col[i] = (int(_dict['last_report']) - int(_dict['first_report']))
            precip_col[i] = _dict['total']

    df[_accum_label] = precip_col  # precip data column
    df['EPOCH_TIMEDELTA'] = epoch_delta_col  # used to filter out erroneous time periods

    df = df[abs(df['EPOCH_TIMEDELTA'] - delta) < .0005 * delta]
    df = df[['STID', _accum_label]].set_index('STID', drop=True)

    return df


# TODO use metadata service to more quickly retrieve the nearest stations then retrieve data for the the k stations.
# Should be faster
def precip_meso_knn(m: Meso, start: str, end: str, stid: str, k: int = 5, **kwargs) -> pandas.DataFrame:
    '''
    Returns a pandas dataframe of the k nearest neighbor stations to the given 'stid' of the same observation using the
    Synoptic Data API radius argument.
    :param m: Meso instance for retrieving data.
    :param start: start of observation window.
    :param end: end of observation window.
    :param stid: station id to find k nearest neighbors of.
    :param k: the number of neighbors to get
    :param kwargs: other key word arguments for the API.
    :return: k nearset stations to 'stid' of the same observation
    '''
    if 'pmode' in kwargs and kwargs['pmode'] != 'totals':
        raise ValueError('pmode must be totals')

    kwargs['timeformat'] = '%s'
    kwargs['radius'] = stid + ',50'
    kwargs['limit'] = str(k)

    neighborhood = precip_dataframe(m, start, end, **kwargs)
    return neighborhood
