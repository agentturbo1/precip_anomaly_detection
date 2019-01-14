from pandas.io.json import json_normalize
from datetime import datetime
import numpy as np


def precip_dataframe(m, start, end, **kwargs):
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
