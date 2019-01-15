# SingleAttrDetect
from typing import Callable

import numpy as np
from MesoPy import Meso
from scipy.stats import norm
from statsmodels import robust

from MesoTools.MesoDataframes import precip_dataframe, precip_meso_knn

'''
MSAD (Median Single Attribute Anomaly Detection) Class
A model using a median algorithm that estimates a probability with given a comparison function for each data point.

This model, by default, relies on the Synoptic API to find the nearest neighbors instead of manually calculating knn 
using certain spatial attributes. If spatial column indices are passed then classical knn is calculated.
'''


class PrecipMSAD:
    def __init__(self):
        self.data = None
        self.spatial_ind = None
        self.attr_func = None
        self.comp_func = None
        self.stid_ind = None
        self.comp_values = None

    def fit(self, m: Meso, k: int = 5, attr_func: Callable[[float], float] = lambda x: x,
            comp_func: Callable[[float, float], float] = lambda f, g: f - g, spatial_ind: int = None,
            stid_ind: [int, str] = 'index', attr_ind: int = 0, **kwargs) -> None:
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
            raise NotImplemented('Classical kNN is not implemented yet.')
        else:
            if self.stid_ind == 'index':
                stids = self.data.index.values
            else:
                stids = self.data.iloc[:, self.stid_ind].values

            neighbor_values = np.empty(stids.shape[0])
            attr_values = np.empty(stids.shape[0])
            self.comp_values = np.empty(stids.shape[0])

            # compute neighborhoods, attribute function, neighborhood function, and comp function for each data point
            for i, stid in enumerate(stids):
                knn_i = precip_meso_knn(m, start, end, stid, k=k, **kwargs).values  # neighborhood observations
                neighbor_values[i] = np.median(knn_i)  # neighborhood function
                attr_values[i] = self.attr_func(self.data.iloc[:, attr_ind][i])  # attribute function
                self.comp_values[i] = self.comp_func(attr_values[i], neighbor_values[i])  # comp function

    def detect(self, alpha: float = 0.01):
        mu_star = np.median(self.comp_values)
        sig_star = robust.mad(self.comp_values)

        ys = abs((self.comp_values - mu_star) / sig_star)
        outliers = ys >= abs(norm.ppf(alpha / 2))

        return self.data[outliers]


if __name__ == '__main__':
    m = Meso(token='demotoken')
    df = precip_dataframe(m, '201803010000', '201810140000', pmode='totals', radius='wbb,10', timeformat='%s')

    print(df)

    d = PrecipMSAD()
    d.fit(m, k=8, start='201803010000', end='201810140000', pmode='totals', radius='wbb,10', timeformat='%s')

    print(d.detect())
