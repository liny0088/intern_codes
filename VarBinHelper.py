import os
import pandas as pd
import numpy as np
from time import time as now
from scipy.stats import chisquare

# Chi-Squared Percent Point Function
from scipy.stats import chi2
# define probability
p = 0.95
df = 1
# retrieve value <= probability
CRITICAL_VALUE = chi2.ppf(p, df)
print(CRITICAL_VALUE)   ## the critical value for later Chi2 merging

class VarBinHelper:
    
    ## has attributes : 
        ## label, 
        ## bin_data, 
        ## original_data
        ## chi2records
        ## critical_value
    
    def __init__(self,label):
        self.label = label
        self.set_critical_value(0.95,1)
#         self.bin_data = 0   <-- see if need this

    def set_critical_value(self,p=0.95,df=1):
        self.critical_value = chi2.ppf(p, df)

    def init_equal_frequency(self, x, bin_rate=0.01):  ## bin_rate < 1 means each bin has same proprtion (eg. 0.05) of all samples. 
                                                  ## >1 means each bin has fixed number of samples
        if bin_rate > 1:     ## find the size of bin
            bin_size = int(bin_rate)
        else: 
            bin_size = int(bin_rate*len(x))

        sorted_x = x.sort_values()  ## sort the varibale for later binning
        sorted_x = sorted_x.reset_index(drop=True)

        bin_up=[]
        bin_low =[-np.inf]

        index=bin_size-1

        while index < len(sorted_x):         ##  Jump every <bin_size> in the sorted X array to record cut points
            bin_up.append(sorted_x[index])   ##  every bin_low is exclusive, bin_up is inclusive, interval like (low,up]
            bin_low.append(sorted_x[index])
            index+=bin_size

        bin_low = bin_low[:-1]
        bin_up[-1]= np.inf
        result = pd.DataFrame({'bin_low':bin_low,'bin_up':bin_up})
        result.index.name = 'bin_num'
        
        self.bin_data = result
        return result
    
    def mapping_bin(self, original_data, bin_data = None, label = None):
    
        ## original data should be 2 columns, X and Y, column 0 is X
        
        if label is None:
            label = self.label    
                
        var_name = original_data.columns[0]
        if var_name == label:
            var_name = original_data.columns[1]   ## find the X var name
            
        if bin_data is None:                ## find own attribute, OR run initialise bin, if bin_data is not given
            try:
                bin_data = self.bin_data
            except:
                bin_data = self.init_equal_frequency(original_data[var_name])
                
        outputDF = original_data.copy()
        outputDF['bin'] = 0
        
        for index, row in bin_data.iterrows():  ## Actual mapping
            outputDF.loc[(outputDF[var_name]>row.bin_low) & (outputDF[var_name]<=row.bin_up),'bin'] = index
            
        self.original_data = outputDF   ## update object attribute when finished
        return outputDF
    
    def calc_chi_2(self, bin_data = None, original_data_mapped = None, label = None): 
    ## to generate the first table of chi2 and bad rates etc, for further merging
        if bin_data is None:
            bin_data = self.bin_data
        if original_data_mapped is None:
            original_data_mapped = self.original_data
        if label is None:
            label = self.label    
            
        # bin_data is the output from initialisation (same frequency or same distance)
        # original_data_mapped should have 3 columns, just the X var and Y label, + mapping output

        total_bad = len(original_data_mapped.loc[original_data_mapped[label]==1])
        total_good = len(original_data_mapped.loc[original_data_mapped[label]==0])

        df = pd.DataFrame(columns = ["bin_low","bin_up","sample_count","bad_count","good_count","bad_rate","bad_count_exp","good_count_exp","Chi_2","Chi_2_if_merge"],index=bin_data.index)
        df.loc[:,['bin_low','bin_up']] = bin_data
        
        for index, row in df.iterrows():
            row.sample_count = len(original_data_mapped.loc[(original_data_mapped.bin == index)])
            row.bad_count = len(original_data_mapped.loc[(original_data_mapped.bin == index) & (original_data_mapped[label]==1)])
            row.good_count = len(original_data_mapped.loc[(original_data_mapped.bin == index) & (original_data_mapped[label]==0)])
            row.bad_count_exp = (row.sample_count)/len(original_data_mapped)*total_bad
            row.good_count_exp = (row.sample_count)/len(original_data_mapped)*total_good
            row.Chi_2 = chisquare([row.bad_count,row.good_count], f_exp=[row.bad_count_exp,row.good_count_exp])[0]
            if index>0:
                row.Chi_2_if_merge = row.Chi_2 + df.Chi_2[index-1]
                
        df.bad_rate = df.bad_count / df.sample_count        
        self.chi2records = df
        return df
    
    def merge_2_bins_in_df(self, index, df_in = None):  ## merging row with index and index+1
        if df_in is None:
            df_in = self.chi2records
        ## the df here should follow the output of cal_Chi_2()
        df = df_in.copy()
        total_count = df.sample_count.sum()
        total_bad = df.bad_count.sum()
        total_good = df.good_count.sum()

        row = df.loc[index]
        next_row = df.loc[index+1]

        row.bin_up = next_row.bin_up
        row.sample_count += next_row.sample_count
        row.bad_count += next_row.bad_count
        row.good_count += next_row.good_count    
        row.bad_count_exp  = row.sample_count / total_count*total_bad
        row.good_count_exp  = row.sample_count / total_count*total_good
        row.Chi_2  = chisquare(f_obs=[row.bad_count,row.good_count], f_exp=[row.bad_count_exp,row.good_count_exp])[0]
        if index!=0:
            row.Chi_2_if_merge = row.Chi_2 + df.loc[index-1, 'Chi_2']
        try:
            df.loc[index+2, 'Chi_2_if_merge'] = row.Chi_2 + df.loc[index+2, 'Chi_2']  ## because the second last row does not have index+2 row
        except:
            pass
        df.loc[index+1] = row
        df = df.drop_duplicates().reset_index(drop=True)
        df.bad_rate = df.bad_count / df.sample_count
        return df
    
    def chi2_merge_loop(self, df = None, critical_value=None, min_bins = 2, max_bins = None):
        if df is None:
            copyDF = self.chi2records.copy()
        else:
            copyDF = df.copy()
            
        if critical_value is None:
            critical_value = self.critical_value
        
        while len(copyDF) > min_bins:    ## merge all bins which if merge, will have Chi2 < critical value.
            Chi2_as_num = pd.to_numeric(copyDF['Chi_2_if_merge'])
            index = Chi2_as_num.idxmin()
            if copyDF.loc[index,'Chi_2_if_merge'] > critical_value:
                break
            copyDF = self.merge_2_bins_in_df(index-1, copyDF)

          ### -- below is to merge by first index with Chi2 < critical value  
    #         chi2_if_merge_with_previous = copyDF.loc[row_index,'Chi_2_if_merge']

    #         if (chi2_if_merge_with_previous < critical_value) and (chi2_if_merge_with_previous < copyDF.loc[row_index+1,'Chi_2_if_merge']):
    #             copyDF = merge_2_bins_in_df(copyDF, row_index-1)
    #             row_index = 1
    #         else:
    #             row_index+=1

        if max_bins is not None:    ## further merge bins if there is a required number of bins
            while max_bins<len(copyDF):  
                Chi2_as_num = pd.to_numeric(copyDF['Chi_2_if_merge'])
                index = Chi2_as_num.idxmin()
                copyDF = self.merge_2_bins_in_df(index-1,copyDF)
        self.chi2records = copyDF
        return copyDF
        