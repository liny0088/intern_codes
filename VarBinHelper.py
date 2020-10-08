## New OO Helper with cleaned code
import os
import pandas as pd
import numpy as np
from time import time as now
from scipy.stats import chi2, chisquare
import math
from sklearn.base import BaseEstimator, TransformerMixin


class VarBinHelper(BaseEstimator, TransformerMixin):
    ## fit from sklearn is fit(X, y=None), current version here is fit(X, y), must pass in y

    def __init__(self, **kwargs):
        ## initialise the object with name of label column, bin_rate, min_bin_num
        self.bin_rate = kwargs.get('bin_rate', 0.01)
        self.min_bin = kwargs.get('min_bin', 2)
        self.max_bin = kwargs.get('max_bin', 10)
        self.chimerge_threshold = kwargs.get("chimerge_threshold", chi2.ppf(0.95, 1))
        self.label = kwargs.get('label', "dpd30")
        self._fit = False

    def set_chimerge_threshold(self, p=0.95, df=1):
        self.chimerge_threshold = chi2.ppf(p, df)

    def bin_cont_init(self, sr_feature, y, bin_rate=0.01, **kwargs):
        ## put each outcome as 1 bin, rank by bad_rate, merge small bins with the neighbor with closest bad_rate
        ## assume all cat outcomes are string, including year eg "2020"
        ls_na = kwargs.get('ls_na', ['nan'])
        method = kwargs.get('method', "chi2")
        min_bin_sample = kwargs.get("min_bin_sample", 0)
        missing = kwargs.get("missing", "separate")
        if len(y) != len(sr_feature):
            print('ERROR: len of series x and y are different')
            return

        df = pd.concat([sr_feature, y], axis=1)
        if bin_rate > 1:  ## find the size of bin
            bin_size = int(max(bin_rate, min_bin_sample))
        else:
            bin_size = int(max(bin_rate * len(sr_feature), min_bin_sample))

        ls_unique = sr_feature.unique().tolist()

        df_bin_interval = pd.DataFrame(columns=['bin_cat', 'total', 'total_rate', 'bad', 'bad_rate'],
                                       index=list(range(len(ls_unique))))
        df_bin_interval.bin_cat = ls_unique

        for idx, row in df_bin_interval.iterrows():
            row.bin_cat = [row.bin_cat]
            row.total = df[sr_feature.name].isin(df_bin_interval.loc[idx, 'bin_cat']).sum()
            row.total_rate = row.total / len(sr_feature)
            row.bad = len(df.loc[(df[sr_feature.name].isin(row.bin_cat)) & (df[y.name] == 1)])
            row.bad_rate = row.bad / row.total

        if missing == "separate":
            ## bin of NA is copied first
            ## drop the NA bin then go to merge small bins -- keep the NA as 1 bin
            ls_na_idx = list()
            if np.nan in ls_na:
                ls_na.remove(np.nan)
                ls_na.append('nan')
            ls_na_exist = list(set(ls_na) & set(ls_unique))
            if list(set(ls_na) - set(ls_na_exist)):
                print("NA values ", list(set(ls_na) - set(ls_na_exist)), " not found in ", sr_feature.name)

            for na_value in ls_na_exist:  ## use interscetion because ls_na might not have values not in ls_unique
                ls_na_idx.append(df_bin_interval.loc[df_bin_interval.bin_cat.apply(lambda x: x == [na_value])].index[0])
            df_na_bin = df_bin_interval.loc[ls_na_idx]
            df_bin_interval = df_bin_interval.drop(index=ls_na_idx)

        df_bin_interval = df_bin_interval.sort_values(by=['bad_rate']).reset_index(drop=True)

        ## only called in this method, merge bins without conisdering Chi 2
        def merge_cat_bin(df_bin_interval, idx_left, idx_right):
            bin_left = df_bin_interval.loc[idx_left]
            bin_right = df_bin_interval.loc[idx_right]
            bin_left.bad += bin_right.bad
            bin_left.total += bin_right.total
            bin_left.bad_rate = bin_left.bad / bin_left.total
            bin_left.bin_cat += bin_right.bin_cat
            return df_bin_interval.drop(idx_right).reset_index(drop=True)

        while df_bin_interval.total.min() < bin_size and (method in ['chi2']):
            ## select and merge small bins if method is chi2 ( later can be skipped if other methods dont need merge.)
            idx = df_bin_interval.total.astype(int).idxmin()
            if idx == 0:
                ## left most bin, no choice, merge with right neighbor
                df_bin_interval = merge_cat_bin(df_bin_interval, idx, idx + 1)
            elif idx == len(df_bin_interval) - 1:
                ## right most bin, merge with left
                df_bin_interval = merge_cat_bin(df_bin_interval, idx - 1, idx)
            else:
                bad_rate = df_bin_interval.bad_rate[idx]
                bad_rate_right = df_bin_interval.bad_rate[idx + 1]
                bad_rate_left = df_bin_interval.bad_rate[idx - 1]
                diff_left = bad_rate - bad_rate_left
                diff_right = bad_rate_right - bad_rate
                merge_right = diff_right < diff_left
                df_bin_interval = merge_cat_bin(df_bin_interval, idx - 1 + merge_right, idx + merge_right)

        #         if missing == "separate":
        #             df_bin_interval = pd.concat([bin_missing, df_bin_interval], axis = 0)
        #             df_bin_interval = df_bin_interval.sort_values(by=['bad_rate']).reset_index(drop=True)

        return df_na_bin, df_bin_interval

    def calc_chi2_cat(self, df_bin_interval):
        ## only being called once for each feature, find chi2 the first time.
        ## No kwargs to pass in
        total_count = df_bin_interval.total.sum()
        total_bad = df_bin_interval.bad.sum()
        total_good = total_count - total_bad
        cols = ["bin_cat", "sample_count", "bad_count", "good_count", "bad_rate", "bad_count_exp",
                "good_count_exp", "chi2", "chi2_after_merge_with_left"]
        df = pd.DataFrame(columns=cols)
        df.bin_cat = df_bin_interval.bin_cat
        df.sample_count = df_bin_interval.total
        df.bad_count = df_bin_interval.bad
        df.bad_rate = df_bin_interval.bad_rate

        for index, row in df.iterrows():
            row.good_count = row.sample_count - row.bad_count
            row.bad_count_exp = (row.sample_count) / total_count * total_bad
            row.good_count_exp = (row.sample_count) / total_count * total_good
            row.chi2 = chisquare([row.bad_count, row.good_count], f_exp=[row.bad_count_exp, row.good_count_exp])[0]
            if index > 0:
                row.chi2_after_merge_with_left = row.chi2 + df.chi2[index - 1]

        return df

    def bin_equal_freq(self, sr_feature, **kwargs):

        ## missing value handling --> default is 1 single bin!
        ## bin_rate < 1 means each bin has same proprtion (eg. 0.05) of all samples.
        ## min_bin_sample --> 100  -- > optional, dfaut = 0
        ## prioritise bin_rate --> is must have
        ## if dont fulfill, error
        ## >1 means each bin has fixed number of samples
        ls_na = kwargs.get('ls_na', [np.nan])
        bin_rate = kwargs.get("bin_rate", self.bin_rate)
        min_bin_sample = kwargs.get("min_bin_sample", 0)
        missing = kwargs.get("missing", "separate")

        if bin_rate > 1:  ## find the size of bin
            bin_size = int(max(bin_rate, min_bin_sample))
        else:
            bin_size = int(max(bin_rate * len(sr_feature), min_bin_sample))

        ## sort the varibale for later binning, not using unique values because we are doing same frequency
        sr_feature_sorted = sr_feature.sort_values().reset_index(drop=True)
        ls_bin_up = list()
        ls_bin_low = [-np.inf]  ## first lower bound is -inf
        if missing == "separate":  ## if choose separate bin for missing value, add np.nan as bin 0
            ls_bin_up = [np.nan]
            ls_bin_low = [np.nan, -np.inf]
        idx = bin_size - 1  ## initialise the running index to look at first cut point

        ##  Start Binning
        ##  Jump every <bin_size> in the sorted X array to record cut points
        while idx < len(sr_feature_sorted):
            cur_val = sr_feature_sorted[idx]
            ##  every bin_low is exclusive, bin_up is inclusive, interval like (low,up]
            ## prevent having intervals like (x,x], empty bin
            if cur_val not in ls_bin_up and not math.isnan(cur_val):
                ls_bin_up.append(cur_val)
                ls_bin_low.append(cur_val)

            ## inspect the next value in sr_feature_sorted after <bin_size>
            idx += bin_size

        ## assume the highest bin is small, merge with 2nd highest bin, set upper bound as inf
        ls_bin_low = ls_bin_low[:-1]
        ls_bin_up[-1] = np.inf
        df_bin_interval = pd.DataFrame({'bin_low': ls_bin_low, 'bin_up': ls_bin_up})
        df_bin_interval.index.name = 'bin_num'

        return df_bin_interval

    def map_bin(self, sr_feature, df_bin_interval, **kwargs):

        ## sr_feature data should be 1 column of series-like, or DF
        inplace = kwargs.get("inplace",
                             False)  ## by default will not overwrite sr_feature values, but add a column "bin"
        missing = kwargs.get("missing",
                             "separate")  ## missing value handling -- by default NAs go to bin 0, a unique bin
        cat = kwargs.get('cat', False)

        var_name = sr_feature.name
        df = pd.DataFrame(sr_feature, columns=[var_name])  ## the working df, will be returned
        df['bin'] = 0
        idx_offset = 0
        if missing == "separate" and cat == False:
            df.loc[df[var_name].isna(), 'bin'] = 0
            idx_offset = 1

        ## Actual mapping starts, iterates by interals
        if cat:
            for idx, row in df_bin_interval.iterrows():
                df.loc[(df[var_name].isin(row.bin_cat)), 'bin'] = idx
        else:
            for idx, row in df_bin_interval.iloc[idx_offset:, ].iterrows():
                df.loc[(df[var_name] > row.bin_low) & (df[var_name] <= row.bin_up), 'bin'] = idx

        if inplace:
            df = df.drop(columns=[var_name])
            df.columns = [var_name]
        return df

    def calc_chi2(self, df_mapped, y, df_bin_interval, **kwargs):
        ## deal with both continuous feature, expect X have 2 columns, just the X var + mapping output
        ## df_bin_interval is the output from initialisation (same frequency or same distance)
        label = kwargs.get("label", self.label)
        cat = kwargs.get("cat", False)
        if len(df_mapped) != len(y):
            print('Error: lengths of mapped X and y are different')
            return
        df_mapped = pd.concat([df_mapped, y], axis=1)
        cols = ["bin_low", "bin_up", "sample_count", "bad_count", "good_count", "bad_rate", "bad_count_exp",
                "good_count_exp", "chi2", "chi2_after_merge_with_left"]
        if cat:
            cols = ["bin_cat"] + cols[2:]
        df_mapped = pd.concat([df_mapped, y], axis=1)
        total_bad = len(df_mapped.loc[df_mapped[label] == 1])  ## find the total bad count and good count
        total_good = len(df_mapped.loc[df_mapped[label] == 0])

        ## working df, to be returned
        if cat:
            cat_count = df_bin_interval
            df = pd.DataFrame(columns=cols, index=list(range(cat_count)))
            df.loc[:, ['bin_low', 'bin_up']] = df_bin_interval
        else:
            df = pd.DataFrame(columns=cols, index=df_bin_interval.index)
            df.loc[:, ['bin_low', 'bin_up']] = df_bin_interval

            for index, row in df.iterrows():
                row.sample_count = len(df_mapped.loc[(df_mapped.bin == index)])
            row.bad_count = len(df_mapped.loc[(df_mapped.bin == index) & (df_mapped[label] == 1)])
            row.good_count = len(df_mapped.loc[(df_mapped.bin == index) & (df_mapped[label] == 0)])
            row.bad_count_exp = (row.sample_count) / len(df_mapped) * total_bad
            row.good_count_exp = (row.sample_count) / len(df_mapped) * total_good
            row.chi2 = chisquare([row.bad_count, row.good_count], f_exp=[row.bad_count_exp, row.good_count_exp])[0]
            if index > 0:
                row.chi2_after_merge_with_left = row.chi2 + df.chi2[index - 1]
            if row.sample_count != 0:
                row.bad_rate = row.bad_count / row.sample_count
            else:
                row.bad_rate = np.nan

        return df

    def merge_pair(self, df_chi2, idx_left, idx_right):
        ## merge row with idx_left and idx_right, called by chi2_merge(), both cat and continuous

        df = df_chi2  ## will return this df
        count_toal = df.sample_count.sum()
        bad_total = df.bad_count.sum()
        good_total = df.good_count.sum()

        row = df.loc[idx_left]
        next_row = df.loc[idx_right]

        try:
            row.bin_up = next_row.bin_up  # assign upper interval, continuous
        except:
            row.bin_cat += next_row.bin_cat  # merge list, cat

        row.sample_count += next_row.sample_count
        row.bad_count += next_row.bad_count
        row.good_count += next_row.good_count
        row.bad_count_exp = row.sample_count / count_toal * bad_total
        row.good_count_exp = row.sample_count / count_toal * good_total
        row.chi2 = chisquare(f_obs=[row.bad_count, row.good_count], f_exp=[row.bad_count_exp, row.good_count_exp])[0]

        if row.sample_count != 0:
            row.bad_rate = row.bad_count / row.sample_count
        else:
            row.bad_rate = np.nan

        if idx_left > 0:
            row.chi2_after_merge_with_left = row.chi2 + df.loc[idx_left - 1, 'chi2']  ## the left neighbor of left bin
        if idx_left + 2 < len(df_chi2):
            ## because the second last row does not have index+2 row, update the chi2 if merge with right bin's right neighbor
            df.loc[idx_left + 2, 'chi2_after_merge_with_left'] = row.chi2 + df.loc[idx_left + 2, 'chi2']

        df.loc[idx_left] = row
        return df.drop([idx_right]).reset_index(drop=True)

    def chi2_merge(self, df_chi2, **kwargs):
        ## merge all bins pairs with Chi2 < critical value, starting with lowest Chi 2 value
        ## stop when min_bin is reached, or when no more Chi 2 < critical

        chimerge_threshold = kwargs.get("chimerge_threshold", self.chimerge_threshold)
        min_bin = kwargs.get("min_bin", self.min_bin)
        max_bin = kwargs.get("max_bin", self.max_bin)

        while len(df_chi2) > min_bin:
            sr_chi2 = df_chi2['chi2_after_merge_with_left'][1:]  ## index 0's value is NA, we use index 1 onwards
            idx_min_chi2 = sr_chi2.astype(float).idxmin()
            if df_chi2.loc[idx_min_chi2, 'chi2_after_merge_with_left'] > chimerge_threshold:
                break  ## stop this loop if no more Chi 2 < threshold
            idx_right = idx_min_chi2
            idx_left = idx_min_chi2 - 1
            df_chi2 = self.merge_pair(df_chi2, idx_left, idx_right)

        if max_bin is not None:  ## further merge bins if max_bin < current bin count
            while max_bin < len(df_chi2):
                sr_chi2 = df_chi2['chi2_after_merge_with_left'][1:]
                idx_min_chi2 = sr_chi2.astype(float).idxmin()
                idx_right = idx_min_chi2
                idx_left = idx_min_chi2 - 1
                df_chi2 = self.merge_pair(df_chi2, idx_left, idx_right)

        df_bin_interval = df_chi2.drop(
            columns=["good_count", "bad_count_exp", "good_count_exp", "chi2", "chi2_after_merge_with_left"]).copy()
        df_bin_interval.columns = df_bin_interval.columns.tolist()[:-3] + ["total", 'bad',
                                                                           'bad_rate']  ## handles both cat and continuous
        df_bin_interval['total_rate'] = df_bin_interval.total / df_bin_interval.total.sum()
        cols = df_bin_interval.columns.tolist()[:-4] + ['total', 'total_rate', 'bad', 'bad_rate']
        df_bin_interval = df_bin_interval[cols]

        return df_bin_interval, df_chi2

    def fit_single_cont(self, x, y, **kwargs):
        ## x,y should be pandas dataframr with 1 column each
        label = kwargs.get("label", self.label)
        method = kwargs.get("method", "chi2")
        bin_rate = kwargs.get("bin_rate", 0.01)
        min_bin = kwargs.get("min_bin", self.min_bin)
        max_bin = kwargs.get("max_bin", self.max_bin)
        min_bin_sample = kwargs.get("min_bin_sample", 0)
        missing = kwargs.get("missing", "separate")
        keep_missing = kwargs.get("keep_missing", True)

        if method == "chi2":
            chimerge_threshold = kwargs.get("chimerge_threshold", self.chimerge_threshold)
            print(bin_rate)
            df_bin_interval = self.bin_equal_freq(sr_feature=x, **kwargs)
            df_mapped = self.map_bin(x, df_bin_interval, missing=missing)
            df_chi2 = self.calc_chi2(df_mapped, y, df_bin_interval, label=label)
            df_bin_interval, df_chi2 = self.chi2_merge(df_chi2, **kwargs)

        return df_bin_interval

    def fit_single_cat(self, x, y, **kwargs):
        method = kwargs.get("method", "chi2")
        df_na_bin, df_bin_interval = self.bin_cont_init(x, y, **kwargs)

        if method == 'chi2':
            df_chi2 = self.calc_chi2_cat(df_bin_interval)  ## no need any kwargs
            df_bin_interval, df_chi2 = self.chi2_merge(df_chi2, **kwargs)
            df_bin_interval = pd.concat([df_bin_interval, df_na_bin], axis=0).sort_values(by=['bad_rate']).reset_index(
                drop=True)

        return df_bin_interval

    def fit(self, X, y, **kwargs):
        ls_cat_feature = kwargs.get("ls_cat_feature", [])  ## default assume no cat
        label = kwargs.get("label", self.label)
        df_bin_model = pd.DataFrame(columns=['feature_name', 'cat', 'bin'])
        ls_bin = list()
        ls_var = list()
        ls_iscat = list()  ## will be boolean
        for var_name in ls_cat_feature:
            print(var_name)
            df_bin_interval = self.fit_single_cat(X[var_name].astype(str), y,
                                                  **kwargs)  ## assume all cat is str, also force to str in transform
            ls_bin.append(df_bin_interval)
            ls_var.append(var_name)
            ls_iscat.append(True)
            # df_bin_model.loc[df_bin_model.feature == var_name,'bin'] = [df_bin_interval]

        for var_name in list(set(X.columns.tolist()) - set(ls_cat_feature)):
            print(var_name)
            df_bin_interval = pd.DataFrame(['placeholder'])
            # df_bin_interval = self.fit_single_cont(X[var_name] , y, **kwargs)
            ls_bin.append(df_bin_interval)
            ls_var.append(var_name)
            ls_iscat.append(False)

        df_bin_model['feature_name'] = ls_var
        df_bin_model['cat'] = ls_iscat
        df_bin_model['bin'] = ls_bin
        self.model = df_bin_model
        self._fit = True
        return df_bin_model

    def transform(self, X, **kwargs):
        ## will return a new DF
        inplace = kwargs.get("inplace", False)
        if self._fit == False:
            print("No model exists, please call .fit(X,y) to fit the model first")
            return
        inplace = label = kwargs.get("inplace", False)

        df = pd.DataFrame()
        for idx, feature in self.model.iterrows():
            name = feature['feature_name']
            if feature['cat'] == True:
                df = pd.concat([df, self.map_bin(X[name].astype(str), feature.bin, inplace=inplace, cat=True)], axis=1)
            else:
                pass  # = self.map_bin(X[name], feature.bin)

        return df
