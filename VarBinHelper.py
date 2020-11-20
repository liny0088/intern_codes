## New OO Helper with cleaned code

import os
import pandas as pd
import numpy as np
from time import time as now
from scipy.stats import chi2, chisquare
import math
from sklearn.base import TransformerMixin


class VarBinHelper(TransformerMixin):

    def __init__(self, **kwargs):
        ## initialise the object with name of label column, min_sample, min_bin_num
        self.min_sample = kwargs.get('min_sample', 0.02)
        self.min_bin = kwargs.get('min_bin', 2)
        self.max_bin = kwargs.get('max_bin', 10)
        self.chimerge_threshold = kwargs.get("chimerge_threshold", chi2.ppf(0.95, 1))
        self.label = kwargs.get('label', None)
        self._fit = False
        self.missing_values_found = {}  ## will be a dict

        ## to make the class interface same as other versions from the team
        self.categorical_features = None  ## updated in fit()
        self.numerical_features = None  ## updated in fit()
        self.woe_encoder = None  ## model
        self.dict_binlist = None  ## model.bin_info

    def set_chimerge_threshold(self, p=0.95, df=1):
        self.chimerge_threshold = chi2.ppf(p, df)

    def init_cat_bin(self, sr_feature, y, min_sample=0.01, **kwargs):
        ## put each outcome as 1 bin, rank by bad_rate, merge small bins with the neighbor with closest bad_rate
        ## assume all categorical values are string, including year eg. "2020"
        method = kwargs.get('method', "chi_merge")
        min_bin_size = kwargs.get("min_bin_size", 5)
        multi_missing = kwargs.get("multi_missing", False)
        dict_na = kwargs.get('missing_values', {})
        merge_category = kwargs.get("merge_category", True)
        init_merge_small_bin = kwargs.get('init_merge_small_bin', True)

        feature_name = sr_feature.name
        if type(dict_na) == list:
            lst_na = dict_na
        else:
            lst_na = dict_na.get(feature_name, ['nan', None])

            # decide bin_size (min sample in a bin)
        df = pd.concat([sr_feature, y], axis=1)
        if min_sample > 1:  ## find the size of bin
            bin_size = int(max(min_sample, min_bin_size))
        else:
            bin_size = int(max(min_sample * len(sr_feature), min_bin_size))

        # initialise each value as 1 bin
        lst_unique = sr_feature.unique().tolist()
        df_bin_interval = pd.DataFrame(columns=['bin', 'total', 'total_rate', 'bad', 'bad_rate'],
                                       index=list(range(len(lst_unique))))
        df_bin_interval.bin = lst_unique

        # calculate total, total_rate, bad, bad_rate for each bin
        for idx, row in df_bin_interval.iterrows():
            row.bin = [row.bin]
            row.total = df[sr_feature.name].isin(df_bin_interval.loc[idx, 'bin']).sum()
            row.total_rate = row.total / len(sr_feature)
            row.bad = len(df.loc[(df[sr_feature.name].isin(row.bin)) & (df[y.name] == 1)])
            row.bad_rate = row.bad / row.total

        # separates NA values as unique bins
        if multi_missing is not None:

            ## determine what NA values exist in this series
            if np.nan in lst_na:
                lst_na.remove(np.nan)
            if 'nan' not in lst_na:
                lst_na.append(
                    'nan')  ## because sr_feature is passed in as df['feature_name].astype(str), we can only find "nan"
            if None not in lst_na:
                lst_na.append(None)

            lst_na_exist = list(set(lst_na) & set(
                lst_unique))  ## use set interscetion because lst_na might have values not in lst_unique
            self.missing_values_found[feature_name] = lst_na_exist

            if list(set(lst_na) - set(lst_na_exist)):
                print("NA values ", list(set(lst_na) - set(lst_na_exist)), " not found in ", sr_feature.name)

            # put NA bins' index in list, use .loc() to extract, then drop them from df_bin_interval
            lst_na_idx = list()
            for na_value in lst_na_exist:
                lst_na_idx.append(df_bin_interval.loc[df_bin_interval.bin.apply(lambda x: x == [na_value])].index[0])

            df_na_bin = df_bin_interval.loc[lst_na_idx]

            if multi_missing == False and len(lst_na_exist) > 0:
                df_temp = pd.DataFrame(columns=['bin', 'total', 'total_rate', 'bad', 'bad_rate'])
                df_temp.bin = [lst_na_exist]
                df_temp.total[0] = df_na_bin.total.sum()
                df_temp.bad[0] = df_na_bin.bad.sum()
                df_temp.total_rate = df_temp.total / len(sr_feature)
                df_temp.bad_rate = df_temp.bad / df_temp.total
                df_na_bin = df_temp

            df_bin_interval = df_bin_interval.drop(index=lst_na_idx)

        df_bin_interval = df_bin_interval.sort_values(by=['bad_rate']).reset_index(drop=True)

        # merge small bins < bin_size for certian methods
        if init_merge_small_bin == True or merge_category == True:  # and merge_category == True
            df_bin_interval = self.merge_small_cat_bins(df_bin_interval, bin_size)

        return df_na_bin, df_bin_interval

    def merge_cat_bin(self, df_bin_interval, idx_left, idx_right):
        bin_left = df_bin_interval.loc[idx_left]
        bin_right = df_bin_interval.loc[idx_right]
        bin_left.bad += bin_right.bad
        bin_left.total += bin_right.total
        bin_left.bad_rate = bin_left.bad / bin_left.total
        bin_left.bin += bin_right.bin
        df_bin_interval = df_bin_interval.drop(idx_right).reset_index(drop=True)
        return df_bin_interval

    def merge_small_cat_bins(self, df_bin_interval, bin_size):

        ## choose the best neighbor(left vs right) to merge, based on bad_rate similarity
        while df_bin_interval.total.min() < bin_size:
            idx = df_bin_interval.total.astype(int).idxmin()
            if idx == 0:
                ## left most bin, no choice, merge with right neighbor
                df_bin_interval = self.merge_cat_bin(df_bin_interval, idx, idx + 1)
            elif idx == len(df_bin_interval) - 1:
                ## right most bin, merge with left neighbor
                df_bin_interval = self.merge_cat_bin(df_bin_interval, idx - 1, idx)
            else:
                bad_rate = df_bin_interval.bad_rate[idx]
                bad_rate_right = df_bin_interval.bad_rate[idx + 1]
                bad_rate_left = df_bin_interval.bad_rate[idx - 1]
                diff_left = bad_rate - bad_rate_left
                diff_right = bad_rate_right - bad_rate
                merge_right = diff_right < diff_left  ## True False but used as 1 and 0 in the next line, to decide where to merge
                df_bin_interval = self.merge_cat_bin(df_bin_interval, idx - 1 + merge_right, idx + merge_right)

        return df_bin_interval

    def calc_chi2_cat(self, df_bin_interval):
        ## only being called once for each feature, find chi2 the first time.
        ## No kwargs to pass in
        total_count = df_bin_interval.total.sum()
        total_bad = df_bin_interval.bad.sum()
        total_good = total_count - total_bad

        ## initialise the df to return
        cols = ["bin", "sample_count", "bad_count", "good_count", "bad_rate", "bad_count_exp",
                "good_count_exp", "chi2", "chi2_after_merge_with_left"]
        df = pd.DataFrame(columns=cols)
        df.bin = df_bin_interval.bin
        df.sample_count = df_bin_interval.total
        df.bad_count = df_bin_interval.bad
        df.bad_rate = df_bin_interval.bad_rate

        ## find chi2 related stats for each bin(row)
        for index, row in df.iterrows():
            row.good_count = row.sample_count - row.bad_count
            row.bad_count_exp = (row.sample_count) / total_count * total_bad
            row.good_count_exp = (row.sample_count) / total_count * total_good
            row.chi2 = chisquare([row.bad_count, row.good_count], f_exp=[row.bad_count_exp, row.good_count_exp])[0]
            if index > 0:
                row.chi2_after_merge_with_left = row.chi2 + df.chi2[index - 1]

        return df

    def init_cont(self, sr_feature, y, **kwargs):

        ## missing value handling --> default is 1 single bin!
        ## min_sample < 1 means each bin has same proprtion (eg. 0.05) of all samples.
        ## min_bin_size -->  optional, dfaut = 5
        ## prioritise min_sample --> is must have
        ## if dont fulfill, error
        ## >1 means each bin has fixed number of samples
        min_sample = kwargs.get("min_sample", self.min_sample)
        min_bin_size = kwargs.get("min_bin_size", 5)  # min sample count in a bin
        multi_missing = kwargs.get("multi_missing", False)
        init_method = kwargs.get("init_method", "quantile")
        dict_na = kwargs.get('missing_values', {})
        init_merge_small_bin = kwargs.get('init_merge_small_bin', True)
        feature_name = sr_feature.name
        # sr_feature[sr_feature.isna()] = np.nan ## set all the NAs to np.nan

        if type(dict_na) == list:
            lst_na = dict_na
        else:
            lst_na = dict_na.get(feature_name, [])
            # print("----- 186",lst_na)
        ## find the size of bin
        if min_sample > 1:
            bin_size = int(max(min_sample, min_bin_size))
        else:
            bin_size = int(max(min_sample * len(sr_feature), min_bin_size))

        ## sort the varibale for later binning, not using unique values because we are doing same frequency
        sr_feature_sorted = sr_feature.sort_values().reset_index(drop=True).copy()

        ## if choose separate bin for missing value, add np.nan as a bin, and each value <= -99000 as a bin
        if multi_missing is not None:

            if np.nan not in lst_na:
                lst_na.append(np.nan)
            # print("----- 203",lst_na) # if float("nan") not in lst_na:
            #     lst_na.append(float("nan"))
            array_feature_unique = sr_feature_sorted.unique()
            array_possible_na = array_feature_unique[array_feature_unique <= -990000]  # eg 990001 990003
            # print("----- 207",array_possible_na)

            if dict_na:
                for na_val in array_possible_na.tolist():
                    if na_val not in lst_na:
                        print(na_val, " found in feature:", feature_name, ", but not specified in missing_values.")

            lst_na = list(set(lst_na).union(set(array_possible_na.tolist())))

            lst_na_lst = list()
            for na_value in lst_na:
                if na_value == "nan" or na_value is None:
                    na_value = np.nan
                    if np.nan in lst_na:
                        continue
                else:
                    lst_na_lst.append([na_value])

            sr_feature_sorted = sr_feature_sorted.dropna()
            sr_feature_sorted = sr_feature_sorted[sr_feature_sorted > -990000].reset_index(drop=True)

        ## find the target count of bins for normal bins
        target_bin_count = len(sr_feature_sorted) / bin_size

        idx = bin_size - 1  ## initialise the running index to look at first cut point
        lst_bin_interval = list()
        lst_bin_up = list()
        lst_bin_low = [-990000]  ## first lower bound is -inf

        ## if unique value is smaller than target_bin_count, each is 1 bin
        if (sr_feature_sorted.nunique() < target_bin_count):
            for cur_val in sr_feature_sorted.unique().tolist():
                lst_bin_interval.append(pd.Interval(left=lst_bin_low[-1:][0], right=cur_val, closed='right'))
                lst_bin_up.append(cur_val)
                lst_bin_low.append(cur_val)

        # initialise with equal frequency
        elif init_method == "quantile":
            ##  Start Binning. Jump every <bin_size> in the sorted X array to record cut points
            while idx < len(sr_feature_sorted):
                cur_val = sr_feature_sorted.loc[idx]
                ##  every bin_low is exclusive, bin_up is inclusive, interval like (low,up]
                ## prevent having intervals like (x,x], which is empty bin
                if cur_val in lst_bin_up:
                    ## change idx to point to next new value
                    try:  ## only error is when last unique value count is larger then bin_size
                        idx = sr_feature_sorted[sr_feature_sorted > cur_val].index[0]
                        continue
                    except:
                        pass

                if cur_val not in lst_bin_up:
                    if not math.isnan(cur_val):
                        lst_bin_interval.append(pd.Interval(left=lst_bin_low[-1], right=cur_val, closed='right'))
                        lst_bin_up.append(cur_val)
                        lst_bin_low.append(cur_val)

                ## inspect the next value in sr_feature_sorted after <bin_size>
                idx += bin_size

        # initialise with equal distance
        elif init_method == "step":
            len_sr = len(sr_feature_sorted)
            sr_feature_sorted = sr_feature_sorted[int(0.05 * len_sr): int(0.95 * len_sr)].reset_index(
                drop=True)  ## follow book, ignore < 5% and > 95%
            value_min = sr_feature_sorted[0]
            value_max = sr_feature_sorted[len(sr_feature_sorted) - 1]
            dist = (value_max - value_min) / (len(sr_feature_sorted) / (
                        bin_size * 0.9))  ## (len(sr_feature_sorted) / bin_size) is number of bins to start with
            cur_val = value_min

            # if dist > 0.01:  # round the cut points for simplicity
            #     print("cut ponints will round to 6 dp.")
            #     # dist = round(dist, 6)

            # go through each cut point, add to lists
            while (cur_val < value_max * 1.001):
                if dist > 0.01:
                    cur_val = round(cur_val, 6)
                lst_bin_interval.append(pd.Interval(left=lst_bin_low[-1], right=cur_val, closed='right'))
                lst_bin_up.append(cur_val)
                lst_bin_low.append(cur_val)
                cur_val += dist

            lst_bin_interval.append(pd.Interval(left=lst_bin_low[-1], right=cur_val, closed='right'))
            lst_bin_up.append(value_max)
            lst_bin_low.append(value_max)

        ## assume the highest bin is small, merge with 2nd highest bin, set upper bound as inf
        lst_bin_low = lst_bin_low[:-1]
        lst_bin_up[-1] = np.inf
        lst_bin_interval[-1] = pd.Interval(left=lst_bin_low[-1], right=np.inf, closed='right')

        ## create the df of normal bins to return
        df_bin_interval = pd.DataFrame(columns=['bin', 'bin_low', 'bin_up', 'total', 'total_rate', 'bad', 'bad_rate'])
        df_bin_interval.bin = lst_bin_interval
        df_bin_interval.bin_low = lst_bin_low
        df_bin_interval.bin_up = lst_bin_up
        df_bin_interval.index.name = 'bin_num'

        # calculate 'total', 'total_rate', 'bad', 'bad_rate'
        for idx, row in df_bin_interval.iterrows():
            df_bin_interval.loc[idx, 'total'] = len(sr_feature[(sr_feature > row.bin_low) & (sr_feature <= row.bin_up)])
            df_bin_interval.loc[idx, 'total_rate'] = df_bin_interval.loc[idx, 'total'] / len(sr_feature)
            df_bin_interval.loc[idx, 'bad'] = len(
                y[((sr_feature > row.bin.left) & (sr_feature <= row.bin.right)) & y == 1])
            if df_bin_interval.loc[idx, 'total'] != 0:
                df_bin_interval.loc[idx, 'bad_rate'] = df_bin_interval.loc[idx, 'bad'] / df_bin_interval.loc[
                    idx, 'total']

        ## merge small bins, since equal distance will have empty/small bins
        if init_method == "step" or init_merge_small_bin:
            while (df_bin_interval.total.min() < bin_size):
                idx = df_bin_interval.total.astype(int).idxmin()
                if idx == 0:
                    ## left most bin, no choice, merge with right neighbor
                    df_bin_interval = self.merge_cont_bin(df_bin_interval, idx, idx + 1)
                elif idx == len(df_bin_interval) - 1:
                    ## right most bin, merge with left neighbor
                    df_bin_interval = self.merge_cont_bin(df_bin_interval, idx - 1, idx)
                else:
                    bad_rate = df_bin_interval.bad_rate[idx]
                    bad_rate_right = df_bin_interval.bad_rate[idx + 1]
                    bad_rate_left = df_bin_interval.bad_rate[idx - 1]
                    diff_left = bad_rate - bad_rate_left
                    diff_right = bad_rate_right - bad_rate
                    merge_right = diff_right < diff_left  ## True False but used as 1 and 0 in the next line, to decide where to merge
                    df_bin_interval = self.merge_cont_bin(df_bin_interval, idx - 1 + merge_right, idx + merge_right)

        ## create the df of NA bins
        df_na_bin = pd.DataFrame(columns=['bin', 'total', 'total_rate', 'bad', 'bad_rate'])
        df_na_bin.bin = lst_na_lst
        lst_na_exist = []

        for idx, row in df_na_bin.iterrows():
            row.total = sr_feature.isin(row.bin).sum()
            row.total_rate = row.total / len(sr_feature)
            row.bad = len(y[sr_feature.isin(row.bin) & y == 1])
            if row.total != 0:
                row.bad_rate = row.bad / row.total
                lst_na_exist += row.bin
            elif dict_na:
                print(row.bin[0], ", this missing value does not exist in ", feature_name)

        self.missing_values_found[
            feature_name] = lst_na_exist  # update object attribute, for later checking in transform()

        if multi_missing == False:
            df_temp = pd.DataFrame(columns=['bin', 'total', 'total_rate', 'bad', 'bad_rate'])
            if len(lst_na_exist) == 0:
                lst_na_exist = [np.nan]
            df_temp.bin = [lst_na_exist]
            df_temp.total[0] = df_na_bin.total.sum()
            df_temp.bad[0] = df_na_bin.bad.sum()
            df_temp.total_rate = df_temp.total / len(sr_feature)
            df_temp.bad_rate = df_temp.bad / df_temp.total
            df_na_bin = df_temp

        return df_na_bin, df_bin_interval

    def map_bin(self, sr_feature, df_bin_interval, **kwargs):
        ## maps both categorical and numerical x
        ## sr_feature data should be 1 column of series-like
        inplace = kwargs.get("inplace",
                             False)  ## by default will not overwrite sr_feature values, but add a column "bin"
        cat = kwargs.get('cat', False)
        bin_only = kwargs.get('bin_only', None)

        ## df is to record intermediate, will be returned
        var_name = sr_feature.name
        df = pd.DataFrame(sr_feature, columns=[var_name])
        df[(var_name + '_bin')] = df[var_name]
        lst_bins = []
        cat_count = 0

        ## Mapping starts, iterates by intevals, for categorical, and NA bins of numerical, row.bin is a list, other numerical row.bin is a pd.Interval
        if cat:
            for idx, row in df_bin_interval.iterrows():
                # df.loc[(df[var_name].isin(row.bin)), (var_name+'_bin')] = idx
                if bin_only is None:
                    value = idx
                elif bin_only == True:
                    value = row.bin
                elif bin_only == False:
                    value = row.woe

                df[(var_name + '_bin')] = df[(var_name + '_bin')].replace(row.bin, value)
        else:
            for idx, row in df_bin_interval.iterrows():

                if bin_only is None:
                    value = idx
                elif bin_only == True:
                    value = row.bin
                elif bin_only == False:
                    value = row.woe

                if type(row.bin) == pd.Interval:
                    # df.loc[(df[var_name] > row.bin.left) & (df[var_name] <= row.bin.right), (var_name+'_bin')] = idx
                    df[(var_name + '_bin')] = df[(var_name + '_bin')].mask(
                        ((df[var_name] > float(row.bin.left)) & (df[var_name] <= float(row.bin.right))), value)
                    lst_bins.append(row.bin.left)

                else:
                    # df.loc[(df[var_name].isin(row.bin)), (var_name+'_bin')] = idx
                    df[(var_name + '_bin')] = df[(var_name + '_bin')].replace(row.bin, value)
                    cat_count += 1

            # lst_bins.append(np.inf)
            # df[(var_name+'_bin')] = pd.cut(df[(var_name+'_bin')], bins = lst_bins, labels=False, right=True)+cat_count

        if inplace:
            df = df.drop(columns=[var_name])
            df.columns = [var_name]

        return df

    def calc_chi2(self, df_mapped, y, df_bin_interval, **kwargs):
        ## deal with both continuous feature, expect X have 2 columns, just the X var + mapping output
        ## df_bin_interval is the output from initialisation (same frequency or same distance)
        label = kwargs.get("label", self.label)
        var_name = df_mapped.columns[0]
        df_mapped = pd.concat([df_mapped, y], axis=1)
        df_mapped.columns = [var_name, label]
        cols = ["bin", "bin_low", "bin_up", "sample_count", "bad_count", "good_count", "bad_rate", "bad_count_exp",
                "good_count_exp", "chi2", "chi2_after_merge_with_left"]

        total_bad = df_mapped[label].sum()  ## find the total bad count and good count
        total_good = len(df_mapped) - total_bad

        ## working df, to be returned
        df = pd.DataFrame(columns=cols, index=df_bin_interval.index.astype(int))
        starting_idx = df_bin_interval.index.astype(int).min()
        df.loc[:, ["bin", 'bin_low', 'bin_up']] = df_bin_interval.loc[:, ["bin", 'bin_low', 'bin_up']]

        for idx, row in df.iterrows():
            row.sample_count = len(df_mapped.loc[(df_mapped[var_name] == idx)])
            row.bad_count = len(df_mapped.loc[(df_mapped[var_name] == idx) & (df_mapped[label] == 1)])
            row.good_count = len(df_mapped.loc[(df_mapped[var_name] == idx) & (df_mapped[label] == 0)])
            row.bad_count_exp = (row.sample_count) / len(df_mapped) * total_bad
            row.good_count_exp = (row.sample_count) / len(df_mapped) * total_good
            row.chi2 = chisquare([row.bad_count, row.good_count], f_exp=[row.bad_count_exp, row.good_count_exp])[0]
            if idx > starting_idx:
                row.chi2_after_merge_with_left = row.chi2 + df.chi2[idx - 1]
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
            row.bin = pd.Interval(left=row.bin.left, right=row.bin_up, closed='right')
        except:
            row.bin += next_row.bin  # merge list, cat

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

        if idx_left > df.index.min():
            row.chi2_after_merge_with_left = row.chi2 + df.loc[idx_left - 1, 'chi2']  ## the left neighbor of left bin
        if idx_left + 2 < len(df_chi2):
            ## because the second last row does not have index+2 row, update the chi2 if merge with right bin's right neighbor
            df.loc[idx_left + 2, 'chi2_after_merge_with_left'] = row.chi2 + df.loc[idx_left + 2, 'chi2']

        df.loc[idx_left] = row
        return df.drop([idx_right]).reset_index(drop=True)

    def chi2_merge(self, df_chi2, **kwargs):

        chimerge_threshold = kwargs.get("chimerge_threshold", self.chimerge_threshold)
        min_bin = kwargs.get("min_bin", self.min_bin)
        max_bin = kwargs.get("max_bin", self.max_bin)
        ## merge all bins pairs with chi2 < chimerge_threshold, starting with lowest chi1 value
        ## stop when min_bin is reached, or when no more chi2 < critical
        while len(df_chi2) > min_bin:
            sr_chi2 = df_chi2['chi2_after_merge_with_left'][1:]  ## index 0's value is NA, we use index 1 onwards
            idx_min_chi2 = sr_chi2.astype(float).idxmin()
            if df_chi2.loc[idx_min_chi2, 'chi2_after_merge_with_left'] > chimerge_threshold:
                break  ## stop this loop if no more chi2 < threshold
            idx_right = idx_min_chi2
            idx_left = idx_min_chi2 - 1
            df_chi2 = self.merge_pair(df_chi2, idx_left, idx_right)

        ## further merge bins if max_bin < current bin count
        if max_bin is not None:
            while max_bin < len(df_chi2):
                sr_chi2 = df_chi2['chi2_after_merge_with_left'][1:]
                idx_min_chi2 = sr_chi2.astype(float).idxmin()
                idx_right = idx_min_chi2
                idx_left = idx_min_chi2 - 1
                df_chi2 = self.merge_pair(df_chi2, idx_left, idx_right)

        df_bin_interval = df_chi2.drop(
            columns=["good_count", "bad_count_exp", "good_count_exp", "chi2",
                     "chi2_after_merge_with_left"]).copy()  ## chi2 intermediate workings are dropped
        df_bin_interval.columns = df_bin_interval.columns.tolist()[:-3] + ["total", 'bad',
                                                                           'bad_rate']  ## handles both cat and continuous
        df_bin_interval['total_rate'] = df_bin_interval.total / df_bin_interval.total.sum()
        cols = df_bin_interval.columns.tolist()[:-4] + ['total', 'total_rate', 'bad',
                                                        'bad_rate']  ## re-order the columns
        df_bin_interval = df_bin_interval[cols].reset_index(drop=True)

        return df_bin_interval, df_chi2

    def find_cut_point(self, df_bin_interval, bin_num_temp, **kwargs):
        ## df_bin_temp is df_bin_interval after adding columns in self.top_down_cut()
        method = kwargs.get("method", "iv")
        df_bin_temp = df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp]
        if len(df_bin_temp) == 1:
            return -1, -1

        best_cut_right = -1
        score_best = -1

        if method == 'iv':
            ## try all cut points within the rows in df
            # if 5 bins 0,1,2,3,4 will try cut at 1,2,3,4 bin < cut_point is bin_left.
            # eg cut_point is 2, left is 0,1, right is 2,3,4

            iv_best = -1
            eps = np.finfo(np.float32).eps

            for cut_point in range(df_bin_temp.index.min() + 1, df_bin_temp.index.max()):
                bin_left = df_bin_temp.loc[:cut_point - 1, :]
                bin_right = df_bin_temp.loc[cut_point:, :]
                # represent the parts in WOE in variables
                good_over_good_total_left = (bin_left.total.sum() - bin_left.bad.sum()) / (
                            df_bin_temp.total.sum() - df_bin_temp.bad.sum())
                good_over_good_total_right = (bin_right.total.sum() - bin_right.bad.sum()) / (
                            df_bin_temp.total.sum() - df_bin_temp.bad.sum())
                bad_over_bad_total_left = bin_left.bad.sum() / df_bin_temp.bad.sum()
                bad_over_bad_total_right = bin_right.bad.sum() / df_bin_temp.bad.sum()

                ## to give a very high value when good_over_good_total = 0
                woe_left = np.log((bad_over_bad_total_left + eps) / (good_over_good_total_right + eps))
                woe_right = np.log((bad_over_bad_total_right + eps) / (good_over_good_total_right + eps))

                ## left side iv
                iv = (bad_over_bad_total_left - good_over_good_total_left) * woe_left

                ## right side iv
                iv = iv + (bad_over_bad_total_right - good_over_good_total_right) * woe_right
                if iv > iv_best:
                    iv_best = iv
                    best_cut_right = cut_point

            score_best = iv_best

        if method == "chi":

            chi2_best = -1
            eps = np.finfo(np.float32).eps
            overall_bad_rate = df_bin_temp.bad.sum() / df_bin_temp.total.sum()
            overall_good_rate = 1 - overall_bad_rate

            for cut_point in range(df_bin_temp.index.min() + 1, df_bin_temp.index.max()):
                bin_left = df_bin_temp.loc[:cut_point - 1, :]
                bin_right = df_bin_temp.loc[cut_point:, :]
                # represent the parts in WOE in variables

                expected_bad_left = bin_left.total.sum() * overall_bad_rate
                expected_good_left = bin_left.total.sum() * overall_good_rate
                good_left = bin_left.total.sum() - bin_left.bad.sum()
                # chi2_left =  ( (bin_left.bad.sum() - expected_bad_left)**2 / expected_bad_left ) + ( (good_left - expected_good_left)**2 / expected_good_left )
                chi2_left = chisquare([bin_left.bad.sum(), good_left], f_exp=[expected_bad_left, expected_good_left])[0]

                ## + eps
                expected_bad_right = bin_right.total.sum() * overall_bad_rate
                expected_good_right = bin_right.total.sum() * overall_good_rate
                good_right = bin_right.total.sum() - bin_right.bad.sum()
                # chi2_right =  ( (bin_right.bad.sum() - expected_bad_right)**2 / expected_bad_right ) + ( (good_right - expected_good_right)**2 / expected_good_right )
                chi2_right = \
                chisquare([bin_right.bad.sum(), good_right], f_exp=[expected_bad_right, expected_good_right])[0]
                chi2_total = chi2_left + chi2_right
                # print("line 477 debug chi2_cut, chi2 is ",chi2_total, ", cut point is ",cut_point )

                if chi2_total > chi2_best:
                    chi2_best = chi2_total
                    best_cut_right = cut_point

            ## update best score
            score_best = chi2_best

        if method == "entropy":
            ent_best = -1
            overall_bad_rate = df_bin_temp.bad.sum() / df_bin_temp.total.sum()
            overall_good_rate = 1 - overall_bad_rate
            # entropy_total true for all cuts
            total_sample = df_bin_temp.total.sum()
            entropy_total = 0 - overall_bad_rate * (np.log(overall_bad_rate)) - overall_good_rate * (
                np.log(overall_good_rate))

            for cut_point in range(df_bin_temp.index.min() + 1, df_bin_temp.index.max()):
                bin_left = df_bin_temp.loc[:cut_point - 1, :]
                bin_right = df_bin_temp.loc[cut_point:, :]

                bad_rate_left = bin_left.bad.sum() / bin_left.total.sum()
                good_rate_left = 1 - bad_rate_left
                total_rate_left = bin_left.total.sum() / total_sample

                bad_rate_right = bin_right.bad.sum() / bin_right.total.sum()
                good_rate_right = 1 - bad_rate_right
                total_rate_right = bin_right.total.sum() / total_sample

                entropy_conditinal = 0

                entropy_temp_left = 0
                entropy_temp_left -= good_rate_left * np.log(good_rate_left)
                entropy_temp_left -= bad_rate_left * np.log(bad_rate_left)
                entropy_conditinal = entropy_conditinal + total_rate_left * entropy_temp_left

                entropy_temp_right = 0
                entropy_temp_right -= good_rate_right * np.log(good_rate_right)
                entropy_temp_right -= bad_rate_right * np.log(bad_rate_right)
                entropy_conditinal = entropy_conditinal + total_rate_right * entropy_temp_right

                entropy_cut = 1 - (entropy_conditinal / entropy_total)
                # print("line 519 debug entropy ---- entropy_cut is ", entropy_cut, ", cut point is ",cut_point, " ent cond and ent total is: ", entropy_conditinal, entropy_total )

                if entropy_cut > ent_best:
                    ent_best = entropy_cut
                    best_cut_right = cut_point

            ## update best score
            score_best = ent_best

        return best_cut_right, score_best

    def cut_and_evaluate(self, df_bin_interval, bin_num_temp, **kwargs):
        ## df_bin_temp is df_bin_interval after adding columns in self.top_down_cut()
        method = kwargs.get("method", "iv")
        force_cut = kwargs.get("force_cut", False)
        best_cut_right, score = self.find_cut_point(df_bin_interval, bin_num_temp, **kwargs)
        df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp, "max_score_if_cut"] = score

        # decide wether to cut based on score and method
        old_score = df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp, "score"].iloc[0]
        decide_cut_iv_entropy = ((method == "iv" or method == "entropy") and score > old_score)
        decide_cut_chi2 = (method == "chi" and score > self.chimerge_threshold)  ## and score > old_score ??

        # score better than before, will cut into 2 parts
        if decide_cut_iv_entropy or decide_cut_chi2 or force_cut:
            df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp, "score"] = score

            idx_min = df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp].index.min()
            idx_max = df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp].index.max()

            df_bin_interval.loc[idx_min:best_cut_right - 1, "bin_temp"] = bin_num_temp * 2 + 1
            df_bin_interval.loc[best_cut_right:idx_max, "bin_temp"] = bin_num_temp * 2 + 2
            # print("cutting bin ", bin_num_temp," cut at ", best_cut_right, "score ", score )

        # score no improvement, stop cutting this branch
        else:
            df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp, "keep_cutting"] = 0
            # print("stop cutting bin ", bin_num_temp," sum of keep_cutting is ", df_bin_interval.keep_cutting.sum() )
            # score less than before, stop cutting for this temp bin

        return df_bin_interval

    def cut_top_down(self, df_bin_interval, **kwargs):

        max_bin = kwargs.get("max_bin", self.max_bin)
        min_bin = kwargs.get("min_bin", self.min_bin)
        # pretend that all bins are in the same initial temp bin 0
        df_bin_interval["bin_temp"] = 0
        df_bin_interval["score"] = 0
        df_bin_interval["max_score_if_cut"] = 0
        df_bin_interval["keep_cutting"] = 1

        keep_cutting = (df_bin_interval["keep_cutting"].sum() > 0)

        # start looking at each temp bin and cut
        while keep_cutting:

            # find unique temp bins
            lst_current_bins = df_bin_interval.bin_temp.unique().tolist()
            for bin_num_temp in lst_current_bins:

                ## only try cutting if this temp bin is labelled keep_cutting == 1
                if df_bin_interval.loc[df_bin_interval.bin_temp == bin_num_temp, "keep_cutting"].sum() > 0:
                    # print("cutting bin ", bin_num_temp)
                    df_bin_interval = self.cut_and_evaluate(df_bin_interval, bin_num_temp, **kwargs)

            has_bin_to_cut = (df_bin_interval["keep_cutting"].sum() > 0)
            below_max_bin = (df_bin_interval['bin_temp'].nunique() < max_bin)
            keep_cutting = (has_bin_to_cut and below_max_bin)

        while (df_bin_interval['bin_temp'].nunique() < min_bin):
            # if min bin is not satisfied, keep cutting the highest score possible bin
            idx = df_bin_interval.max_score_if_cut.idxmax()
            bin_num_temp = df_bin_interval.bin_temp[idx]
            # print("to satisfy min bin, force cutting temp bin: ",bin_num_temp)
            df_bin_interval = self.cut_and_evaluate(df_bin_interval, bin_num_temp, force_cut=True, **kwargs)

        # merge the temp bins, using pandas aggregate methods
        df_group = pd.DataFrame(columns=["bin", "total", "total_rate", "bad", "bad_rate"])
        df_aggregate = df_bin_interval.groupby(by=["bin_temp"])["total", "bad"].sum().loc[:,
                       ["total", "bad"]].reset_index(drop=True)
        df_group.total = df_aggregate.total
        df_group.bad = df_aggregate.bad
        total_sample = df_group.total.sum()
        df_group.total_rate = df_group.total / total_sample
        df_group.bad_rate = df_group.bad / df_group.total

        # find the right intervals for each temp bin
        ar_left = df_bin_interval.groupby(by=["bin_temp"])["bin_low"].min().tolist()
        ar_right = df_bin_interval.groupby(by=["bin_temp"])["bin_up"].max().tolist()
        for idx, row in df_group.iterrows():
            df_group.bin[idx] = pd.Interval(left=ar_left[idx], right=ar_right[idx], closed="right")

        # sort by bin interval, min to max
        df_group = df_group.sort_values(by=['bin']).reset_index(drop=True)

        return df_group  ## debug only, actual is df_group

    def set_significant_figures(self, sr_feature, unique_range):  # eg (1000,5000)

        if (len(sr_feature.unique()) < unique_range[1]):
            return sr_feature

        decimal_place = 6  # start trying with round(sr_feature, 6)
        sr_result = sr_feature.copy()

        while (len(sr_result.unique()) > unique_range[1]):
            decimal_place -= 1
            sr_result = round(sr_feature, decimal_place)

        if (len(sr_result.unique()) < unique_range[0]):
            decimal_place += 1
            sr_result = round(sr_feature, decimal_place)

        print(sr_feature.name, " rounded to decimal point: ", decimal_place, "   unique values counts = ",
              len(sr_result.unique()))

        return sr_result

    def find_turn_count(self, sr):
        ## function to find the longest monotonically decreasing / increasing bad rates in a list
        ## called by self.force_monotone()
        turn_count = 0
        if len(sr) <= 2:
            return 0

        for idx in range(1, len(sr) - 1):
            # if it is a turning point
            if (sr[idx - 1] > sr[idx] and sr[idx] < sr[idx + 1]) or (sr[idx - 1] < sr[idx] and sr[idx] > sr[idx + 1]):
                turn_count += 1

        return turn_count

    def merge_cont_bin(self, df_bin_interval, idx_left, idx_right):
        ## simply merging 2 bins without calculating chi2. Useful to handle df_bin_interval after chi2 or other method
        ## called by force_monotone()

        df_copy = df_bin_interval.copy()

        df_copy.loc[idx_left, "bad"] += df_copy.loc[idx_right, "bad"]
        df_copy.loc[idx_left, "total"] += df_copy.loc[idx_right, "total"]
        if 'bin_up' in df_copy.columns.to_list():
            df_copy.loc[idx_left, "bin_up"] = df_copy.loc[idx_right, "bin_up"]
        df_copy.loc[idx_left, "bad_rate"] = df_copy.loc[idx_left, "bad"] / df_copy.loc[idx_left, "total"]
        df_copy.loc[idx_left, "bin"] = pd.Interval(left=df_copy.loc[idx_left, "bin"].left,
                                                   right=df_copy.loc[idx_right, "bin"].right, closed='right')
        df_copy = df_copy.drop(idx_right).reset_index(drop=True)

        return df_copy

    def choose_turning_point_and_neighbor(self, sr_bad_rate):
        ## called by force_monotone()
        ## sr_bad_rate can also be a pd series
        idx_res_left = 0
        idx_res_right = 1
        min_diff = 1  ## bad_rate is 0~1
        idx_turn = 1

        # find the indexes of the pair with closest bad_rate (from turning points)
        for idx in range(1, len(sr_bad_rate) - 1):
            is_up_turn = (sr_bad_rate[idx - 1] > sr_bad_rate[idx] and sr_bad_rate[idx] < sr_bad_rate[idx + 1])
            is_down_turn = (sr_bad_rate[idx - 1] < sr_bad_rate[idx] and sr_bad_rate[idx] > sr_bad_rate[idx + 1])
            # if it is a turning point (first occurance)
            if is_up_turn or is_down_turn:
                diff_left = abs(sr_bad_rate[idx - 1] - sr_bad_rate[idx])
                diff_right = abs(sr_bad_rate[idx] - sr_bad_rate[idx + 1])
                # if difference with left is lower
                if diff_left <= diff_right and diff_left < min_diff:
                    idx_res_left, idx_res_right = idx - 1, idx
                    min_diff = diff_left
                    idx_turn = idx
                # if difference with right is lower
                elif diff_left > diff_right and diff_right < min_diff:
                    idx_res_left, idx_res_right = idx, idx + 1
                    min_diff = diff_right
                    idx_turn = idx

        # return is outside for loop, to find the minimum of bad_rate differences
        return idx_res_left, idx_res_right, idx_turn

    def force_monotone(self, df_bin_interval, **kwargs):
        ## df_bin_interval has columns bin, total, total_rate, bad, bad_rate
        force_mono = kwargs.get('force_mono', 'u_shape')  ## possible values: 'u_shape', 'mono'
        max_bin = kwargs.get('max_bin', 10)  ## need to

        if force_mono == 'u_shape':
            allowed_turns = 1
        else:
            allowed_turns = 0

        ## start merging until allowed number of turning points are reached
        while (self.find_turn_count(df_bin_interval.bad_rate) > allowed_turns):
            # find the left and right index to merge
            idx_left, idx_right, idx_turn = self.choose_turning_point_and_neighbor(df_bin_interval.bad_rate)

            # print(self.find_turn_count(df_bin_interval.bad_rate)," turns, merging ",idx_left," ", idx_right)

            # merge the twn bins
            df_bin_interval = self.merge_cont_bin(df_bin_interval, idx_left, idx_right)

        # if u shape and has 1 last turn
        if force_mono == 'u_shape' and self.find_turn_count(df_bin_interval.bad_rate) == 1:
            idx_left, idx_right, idx_turn = self.choose_turning_point_and_neighbor(df_bin_interval.bad_rate)

            # if that turn is at beginning or end
            if idx_turn == 1 or idx_turn == len(df_bin_interval.bad_rate) - 2:

                # we will merge that with neighbors, untill we see mono
                while (self.find_turn_count(df_bin_interval.bad_rate) > 0):
                    # find the left and right index to merge
                    idx_left, idx_right, idx_turn = self.choose_turning_point_and_neighbor(df_bin_interval.bad_rate)

                    # print(self.find_turn_count(df_bin_interval.bad_rate)," turns, merging ",idx_left," ", idx_right)

                    # merge the twn bins
                    df_bin_interval = self.merge_cont_bin(df_bin_interval, idx_left, idx_right)

        return df_bin_interval

    def calc_woe(self, df_bin_interval):
        eps = np.finfo(float).eps
        total_count = df_bin_interval.total.sum()
        total_bad = df_bin_interval.bad.sum()
        total_good = total_count - total_bad

        df_bin_interval['good'] = df_bin_interval['total'] - df_bin_interval['bad']
        df_bin_interval['good_density'] = df_bin_interval['good'] / total_good
        df_bin_interval['bad_density'] = df_bin_interval['bad'] / total_bad
        df_bin_interval['woe'] = np.log((df_bin_interval['good_density'].astype('float64') + eps) / (
                    df_bin_interval['bad_density'].astype('float64') + eps))
        df_bin_interval['iv'] = np.log((df_bin_interval['good_density'].astype('float64') + eps) / (
                    df_bin_interval['bad_density'].astype('float64') + eps)) * (
                                            df_bin_interval['good_density'].astype('float64') - df_bin_interval[
                                        'bad_density'].astype('float64'))

        return df_bin_interval.drop(columns=['good', 'good_density', 'bad_density'])

    def fit_single_cont(self, x, y, **kwargs):
        method = kwargs.get("method", "iv")
        force_mono = kwargs.get("force_mono", None)
        max_bin = kwargs.get("max_bin", self.max_bin)

        df_na_bin, df_bin_interval = self.init_cont(sr_feature=x, y=y, **kwargs)

        if (df_bin_interval.shape[0] < max_bin):
            ## if bin count after init < max_bin, skip the merging / cutting
            df_bin_interval = df_bin_interval.drop(columns=['bin_low', 'bin_up'])
            print(x.name, "has limited unique values, count < max_bin, skipped merging / cutting")

        elif method == "chi_merge":
            # bottum up merging
            df_all_bin = pd.concat([df_na_bin, df_bin_interval], axis=0).reset_index(drop=True)
            df_mapped = self.map_bin(x, df_all_bin, inplace=True)  ## initial map to both NA and normal bins
            df_chi2 = self.calc_chi2(df_mapped, y, df_all_bin[len(df_na_bin):], **kwargs)
            df_bin_interval, df_chi2 = self.chi2_merge(df_chi2, **kwargs)
            df_bin_interval = df_bin_interval.drop(columns=['bin_low', 'bin_up'])

        else:
            # cutting by iv, chi2, or entropy
            df_bin_interval = self.cut_top_down(df_bin_interval, **kwargs)  # high level method of top down cutting

        # post processing to find total rate
        total_sample = df_bin_interval.total.sum() + df_na_bin.total.sum()
        df_bin_interval.total_rate = df_bin_interval.total / total_sample

        # force monotone of bad rate
        if force_mono:
            df_bin_interval = self.force_monotone(df_bin_interval, force_mono=force_mono)

        ## final merge with NA bins
        df_bin_interval = pd.concat([df_bin_interval, df_na_bin], axis=0).reset_index(
            drop=True)  ## final merge with NA bins

        ## calculate woe and iv of each bin
        df_bin_interval = self.calc_woe(df_bin_interval)

        # drop bins where total count == 0
        dict_na = kwargs.get('missing_values', [])
        if type(dict_na) == list:
            df_bin_interval = df_bin_interval[df_bin_interval.total != 0].reset_index(drop=True)

        return df_bin_interval

    def fit_single_cat(self, x, y, **kwargs):
        ## expects x as a series object like df.column or df['column']
        method = kwargs.get("method", "chi_merge")
        max_bin = kwargs.get("max_bin", self.max_bin)
        merge_category = kwargs.get("merge_category", True)
        ## initialise the bins
        df_na_bin, df_bin_interval = self.init_cat_bin(x, y, **kwargs)

        if (df_bin_interval.shape[0] < max_bin):
            ## if bin count after init < max_bin, skip the merging / cutting
            print(x.name, "has limited unique values, count < max_bin, skipped merging")

        elif merge_category == False:
            print(x.name, "is categorical, not merging bins according to user's input merge_category")

        else:
            # if method == 'chi_merge':   ## Cat has only chi_merge
            df_chi2 = self.calc_chi2_cat(df_bin_interval)
            df_bin_interval, df_chi2 = self.chi2_merge(df_chi2, **kwargs)

        if merge_category:
            # merge categorical bins that have the same bad rates
            while (df_bin_interval.bad_rate.nunique() < len(df_bin_interval)):
                ## to find 2 bins that are equal in bad rate, merge
                df_same_badrate = df_bin_interval.groupby("bad_rate").filter(lambda x: len(x) > 1)
                idx_left = df_same_badrate.index[0]
                idx_right = df_same_badrate.index[1]
                df_bin_interval = self.merge_cat_bin(df_bin_interval, idx_left, idx_right)

        df_bin_interval = pd.concat([df_na_bin, df_bin_interval], axis=0)

        # post processing
        total_sample = df_bin_interval.total.sum()
        df_bin_interval.total_rate = df_bin_interval.total / total_sample

        df_bin_interval = df_bin_interval.sort_values(by=['bad_rate']).reset_index(drop=True)

        df_bin_interval = self.calc_woe(df_bin_interval)

        # drop bins where total count == 0, when missing_values is list
        dict_na = kwargs.get('missing_values', [])
        if type(dict_na) == list:
            df_bin_interval = df_bin_interval[df_bin_interval.total != 0].reset_index(drop=True)

        return df_bin_interval

    def fit(self, df_feature, df_label, **kwargs):
        # note: df_label is y in sklearn, it is a series like df.dpd30

        lst_cat_feature = kwargs.get("categorical_features", [])  ## default assume 0 categorical features
        label = kwargs.get("label", df_label.name)
        self.label = label
        unique_range = kwargs.get("unique_range", None)
        feature_list = kwargs.get("feature_list", df_feature.columns.tolist())
        lst_excluded_ft = kwargs.get("exclude", [])
        dict_na = kwargs.get("missing_values", {})

        self.numerical_features = list(set(df_feature.columns.tolist()) - set(lst_cat_feature) - set([label]))
        self.categorical_features = lst_cat_feature
        if len(lst_cat_feature) == 0:
            print("no categorical_features list is passed, assuming all features are numerical.")

        lst_bin = list()
        lst_ft = list()
        lst_iscat = list()  ## A list of boolean values, storing if a feature is categorical

        ## fit features that are categorical
        for feature_name in lst_cat_feature:
            if feature_name not in df_feature.columns.to_list():
                print("------- ", feature_name,
                      " in param lst_cat_feature NOT found in Dataframe columns, skipped, please check ----------")
                continue
            if (len(df_feature[feature_name]) != len(df_label)):
                print(
                    "fit() skipped for this feature. Please make sure length of x and y are the same for x feature name: ",
                    feature_name)
                continue
            if (feature_name == label) or (feature_name in lst_excluded_ft) or (feature_name not in feature_list):
                continue

            print("------- fitting: ", feature_name, " -------")
            ## assume all categorical value is str, also force to str in self.transform()
            sr_x = df_feature[feature_name].astype(str)
            df_bin_interval = self.fit_single_cat(sr_x, df_label, **kwargs)
            lst_bin.append(df_bin_interval)
            lst_ft.append(feature_name)
            lst_iscat.append(True)

        ## fit features that are continuous
        for feature_name in self.numerical_features:
            if (feature_name == label) or (feature_name in lst_excluded_ft) or (feature_name not in feature_list):
                continue

            print("------- fitting: ", feature_name, " -------")
            sr_x = df_feature[feature_name]
            if unique_range is not None:
                sr_x = self.set_significant_figures(sr_x, unique_range)
            df_bin_interval = self.fit_single_cont(sr_x, df_label, **kwargs)
            lst_bin.append(df_bin_interval)
            lst_ft.append(feature_name)
            lst_iscat.append(False)

        ## prepare return model
        ## return model: df_bin_model has three columns ['feature_name', 'is_cat', 'bin_info']
        ## feature_name is a list of feature names
        ## is_cat indicates whether the feature is categorical(True) or numerical(False)
        ## bin_info is df with columns [ bin, total, total_rate, bad, bad_rate ]
        df_bin_model = pd.DataFrame(columns=['feature_name', 'is_cat', 'bin_info'])
        df_bin_model['feature_name'] = lst_ft
        df_bin_model['is_cat'] = lst_iscat
        df_bin_model[
            'bin_info'] = lst_bin  ## Each bin in bin_info: if categorical, is a list of string values of that bin OR if numerical, is a pd.interval

        self.model = df_bin_model
        self._fit = True

        # make a copy of fit() result
        self.model_backup = self.model.copy()

        lst_df = []
        for index, row in self.model.iterrows():
            df_tmp = row['bin_info']
            df_tmp['var'] = row['feature_name']
            df_tmp['is_cat'] = row['is_cat']
            lst_df.append(df_tmp)

        self.woe_encoder = pd.concat(lst_df)

        return self

    def transform(self, df_feature, **kwargs):
        inplace = kwargs.get("inplace", True)
        bin_only = kwargs.get("bin_only", True)  ## need to change lower level self.map_bin()
        lst_feature_names = kwargs.get("feature_list", [])  ## default transform all, unless specify the columns
        lst_exclude = kwargs.get("exclude", [])
        dict_na = kwargs.get("missing_values", None)

        if self._fit is False:
            raise ValueError("No model exists, please call self.fit(df_feature, df_label) to fit the model first")

        if dict_na is None:
            print("----- No missing_values list is passed in. -----")

        lst_trans = list()
        for idx, row in self.model.iterrows():

            name = row['feature_name']
            if name in lst_exclude:
                continue

            # will transform this feature if user never input feature_list, or the name is in the feature_list
            if (len(lst_feature_names) == 0) or (name in lst_feature_names):
                print("----- transforming: ", name, " -----")

                if dict_na:
                    lst_missing_found = self.missing_values_found.get(name, [])
                    lst_missing_user = dict_na.get(name, [])
                    if set(lst_missing_user) != set(lst_missing_found):
                        print(name, " missing values found do not match what is passed.")
                        print("Found but not passed:", set(lst_missing_found) - set(lst_missing_user),
                              " Passed but not found: ", set(lst_missing_user) - set(lst_missing_found))

                if row['is_cat'] == True:
                    # transform single categorical feature
                    df_trans = self.map_bin(df_feature[name].astype(str), row.bin_info, inplace=True, bin_only=bin_only,
                                            cat=True)
                    lst_trans.append(df_trans)
                else:
                    df_trans = self.map_bin(df_feature[name], row.bin_info, inplace=True, bin_only=bin_only, cat=False)
                    lst_trans.append(df_trans)

        df = pd.concat(lst_trans, axis=1)
        df_copy = df_feature.copy()
        df_copy.update(df)

        if inplace:
            df_feature.update(df)

        return df_copy

    def evaluate_model_bin_count(self):
        if self._fit is False:
            print("No model yet, please call self.fit() first")
            return

        feature_count = self.model.shape[0]

        lst_bin_count = []
        for idx in range(feature_count):
            bin_count = self.model.bin_info[idx].shape[0]
            lst_bin_count.append(bin_count)

        sr_bin_count = pd.Series(lst_bin_count)

        bin_min = sr_bin_count.min()
        bin_max = sr_bin_count.max()
        bin_mean = sr_bin_count.mean()
        print("min, max, mean of bin count is : ", bin_min, " ", bin_max, " ", bin_mean)

        self.model["bin_count"] = sr_bin_count

        return self.model

    def set_rules(self, dict_rules,
                  data):  ## user has to pass in data (df), in order to re-calculate bad, total, woe and iv

        if self._fit is False:
            print("No model yet, please call self.fit() first")
            return

        if dict_rules == "recover":
            self.model = self.model_backup.copy()
            dict_rules = {}

        for key in dict_rules:

            if key in self.model.feature_name.to_list():

                row_feature = self.model.loc[self.model.feature_name == key]
                feature_is_cat = row_feature.is_cat.iloc[0]
                df_bin_interval_user = pd.DataFrame(columns=['bin', 'total', 'total_rate', 'bad', 'bad_rate'])
                lst_user_bin = dict_rules.get(
                    key)  ## it will be a list of lists for cat / list of integers for continuous
                df_bin_info = row_feature.bin_info.iloc[0]

                if feature_is_cat:

                    lst_cat_values = list()
                    # find all the categorical values of this feature
                    for index, row_bin in df_bin_info.iterrows():
                        lst_cat_values += row_bin.bin

                    lst_user_values = list()
                    # find the set of value that user passed in
                    for lst_one_bin in lst_user_bin:
                        lst_user_values += lst_one_bin

                    lst_values_not_in_dict = list(set(lst_cat_values) - set(lst_user_values))

                    if len(lst_values_not_in_dict) > 0:
                        # append the values that user did not pass in as the last bin
                        lst_user_bin.append(lst_values_not_in_dict)

                    df_bin_interval_user.bin = lst_user_bin

                else:
                    # for continuous expect a list like [0,2,4,6,8,12]
                    # default will not expect user to change NA bins
                    lst_na_bins = df_bin_info.loc[df_bin_info['bin'].map(type) == list, 'bin'].to_list()

                    sr_user_bin = pd.Series(lst_user_bin)

                    # if user passes in something like [[-999900],[-999901,-999902],0,2,4,6,9]
                    if (sr_user_bin.map(type) == list).sum() > 0:
                        # we will overwrite the na bins as he wishes
                        lst_na_bins = sr_user_bin[sr_user_bin.map(type) == list].to_list()
                        lst_user_bin = sr_user_bin[sr_user_bin.map(type) != list].to_list()

                    if -990000 not in lst_user_bin:
                        lst_user_bin = [-990000] + lst_user_bin
                    if np.inf not in lst_user_bin:
                        lst_user_bin.append(np.inf)

                    lst_user_bin.sort()

                    lst_bin_low = lst_user_bin[:-1]
                    lst_bin_up = lst_user_bin[1:]

                    lst_bin_interval = list()

                    for i in range(0, len(lst_bin_low)):
                        bin_interval = pd.Interval(left=lst_bin_low[i], right=lst_bin_up[i], closed='right')
                        lst_bin_interval.append(bin_interval)

                        # merge the NA bins
                    lst_bin_interval = lst_bin_interval + lst_na_bins

                    df_bin_interval_user.bin = lst_bin_interval

                ## re-calculate woe , iv .... if there is df passed in, and the feature is found in data
                if (data is not None) and (key in data.columns.to_list()):
                    sr_feature = data[key]
                    y = data[self.label]
                    df = pd.concat([sr_feature, y], axis=1)

                    for idx, row in df_bin_interval_user.iterrows():

                        if type(row.bin) == list:
                            df_bin_interval_user.loc[idx, 'total'] = df[sr_feature.name].isin(
                                df_bin_interval_user.loc[idx, 'bin']).sum()
                            df_bin_interval_user.loc[idx, 'bad'] = len(
                                df.loc[(df[sr_feature.name].isin(row.bin)) & (df[y.name] == 1)])
                        else:
                            df_bin_interval_user.loc[idx, 'total'] = len(
                                sr_feature[(sr_feature > row.bin.left) & (sr_feature <= row.bin.right)])
                            df_bin_interval_user.loc[idx, 'bad'] = len(
                                y[((sr_feature > row.bin.left) & (sr_feature <= row.bin.right)) & y == 1])

                        df_bin_interval_user.loc[idx, 'total_rate'] = df_bin_interval_user.loc[idx, 'total'] / len(
                            sr_feature)
                        if df_bin_interval_user.loc[idx, 'total'] != 0:
                            df_bin_interval_user.loc[idx, 'bad_rate'] = df_bin_interval_user.loc[idx, 'bad'] / \
                                                                        df_bin_interval_user.loc[idx, 'total']

                    df_bin_interval_user = self.calc_woe(df_bin_interval_user)

                # update the bin_info in model
                self.model.bin_info[row_feature.index[0]] = df_bin_interval_user.copy()
                print("updated bins for ", key, " , the df is now like:")
                print(self.model.bin_info[row_feature.index])
                print("user bin df is:")
                print(df_bin_interval_user)

            # if key not in model.feature_name
            else:
                print("column name ", key, " not in model.feature_name")

        lst_df = []
        for index, row in self.model.iterrows():
            df_tmp = row['bin_info']
            df_tmp['var'] = row['feature_name']
            df_tmp['is_cat'] = row['is_cat']
            lst_df.append(df_tmp)

        self.woe_encoder = pd.concat(lst_df)

        return self

    def drop_empty_missing_bin(self):
        if self._fit is False:
            print("No model yet, please call self.fit() first")
            return

        lst_df = []

        for idx, row in self.model.iterrows():
            row.bin_info = row.bin_info.loc[row.bin_info.total > 0].reset_index(drop=True)
            df_tmp = row['bin_info']
            df_tmp['var'] = row['feature_name']
            df_tmp['is_cat'] = row['is_cat']
            lst_df.append(df_tmp)

        self.woe_encoder = pd.concat(lst_df)