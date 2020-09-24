import os
import pandas as pd
import numpy as np
from time import time as now
from scipy.stats import chisquare
import math

# Chi-Squared Percent Point Function
from scipy.stats import chi2
# define probability
p = 0.95
df = 1
# retrieve value <= probability
CRITICAL_VALUE = chi2.ppf(p, df)
# print(CRITICAL_VALUE)   ## the critical value for later Chi2 merging

class VarBinHelper:
    
    ## has attributes: 
        ## label, <<-- should be consistent throughout 1 dataset
        ## bin_intervals, <<-- bin intervals for 1 var
        ## x_y_pair <<-- this is only 1 var and Y, 2 columns
        ## chi2records  <<-- working dataframe for chi2 merge
        ## critical_value <<-- chi2 merge critical value
        ## original_df <<-- entire DF of continuous variables and label column
    
    def __init__(self,label):    ## initialise the object with name of label column
        self.label = label
        self.set_critical_value(0.95,1)
#         self.bin_intervals = 0   <-- see if need this

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

        while index < len(sorted_x):
            current_x = sorted_x[index]  ##  Jump every <bin_size> in the sorted X array to record cut points
            if current_x not in bin_up and not math.isnan(current_x):   ## prevent having intervals like (x,x], empty bin
                bin_up.append(current_x)   ##  every bin_low is exclusive, bin_up is inclusive, interval like (low,up]
                bin_low.append(current_x)
            index+=bin_size

        bin_low = bin_low[:-1]
        bin_up[-1] = np.inf
        result = pd.DataFrame({'bin_low':bin_low,'bin_up':bin_up})
        result.index.name = 'bin_num'
        
        self.bin_intervals = result
        return result
    
    def mapping_bin(self, x_y_pair = None, bin_intervals = None, label = None):
    
        ## original data should be 2 columns, X and Y, column 0 is X
        
        if label is None:
            label = self.label  
        if x_y_pair is None and self.x_y_pair is not None:
            x_y_pair = self.x_y_pair
                
        var_name = x_y_pair.columns[0]
        if var_name == label:
            var_name = x_y_pair.columns[1]   ## find the X var name
            
        if bin_intervals is None:                ## find own attribute, OR run initialise bin, if bin_intervals is not given
            try:
                bin_intervals = self.bin_intervals
            except:
                bin_intervals = self.init_equal_frequency(x_y_pair[var_name])
                
        outputDF = x_y_pair.copy()
        outputDF['bin'] = 0
        
        for index, row in bin_intervals.iterrows():  ## Actual mapping
            outputDF.loc[(outputDF[var_name]>row.bin_low) & (outputDF[var_name]<=row.bin_up),'bin'] = index
            
        self.x_y_pair = outputDF   ## update object attribute when finished, with columns X, Y and 'bin'
        return outputDF
    
    def calc_chi_2(self, bin_intervals = None, x_y_pair_mapped = None, label = None): 
    ## to generate the first table of 
        if bin_intervals is None:
            bin_intervals = self.bin_intervals
        if x_y_pair_mapped is None:
            x_y_pair_mapped = self.x_y_pair
        if label is None:
            label = self.label
            
        # bin_intervals is the output from initialisation (same frequency or same distance)
        # x_y_pair_mapped should have 3 columns, just the X var and Y label, + mapping output

    #     var_name = x_y_pair_mapped.columns[0]
    #     if var_name == label:
    #         var_name = x_y_pair_mapped.columns[1]   < ---- might not need
        total_bad = len(x_y_pair_mapped.loc[x_y_pair_mapped[label]==1])
        total_good = len(x_y_pair_mapped.loc[x_y_pair_mapped[label]==0])

        df = pd.DataFrame(columns = ["bin_low","bin_up","sample_count","bad_count","good_count","bad_rate","bad_count_exp","good_count_exp","Chi_2","Chi_2_if_merge"],index=bin_intervals.index)
        df.loc[:,['bin_low','bin_up']] = bin_intervals
        
        for index, row in df.iterrows():
            row.sample_count = len(x_y_pair_mapped.loc[(x_y_pair_mapped.bin == index)])
            row.bad_count = len(x_y_pair_mapped.loc[(x_y_pair_mapped.bin == index) & (x_y_pair_mapped[label]==1)])
            row.good_count = len(x_y_pair_mapped.loc[(x_y_pair_mapped.bin == index) & (x_y_pair_mapped[label]==0)])
            row.bad_count_exp = (row.sample_count)/len(x_y_pair_mapped)*total_bad
            row.good_count_exp = (row.sample_count)/len(x_y_pair_mapped)*total_good
            row.Chi_2 = chisquare([row.bad_count,row.good_count], f_exp=[row.bad_count_exp,row.good_count_exp])[0]
            if index>0:
                row.Chi_2_if_merge = row.Chi_2 + df.Chi_2[index-1]
            if row.sample_count !=0:
                row.bad_rate = row.bad_count / row.sample_count     
                   
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
        
        while len(copyDF) > min_bins:                              ## merge all bins pairs with Chi2 < critical value, starting with lowest Chi 2 value
            Chi2_as_num = pd.to_numeric(copyDF['Chi_2_if_merge'])  ## stop when min_bin is reached, or when no more Chi 2 < critical value
            index = Chi2_as_num.idxmin()
            if copyDF.loc[index,'Chi_2_if_merge'] > critical_value:
                break
            copyDF = self.merge_2_bins_in_df(index-1, copyDF)

        if max_bins is not None:                         ## further merge bins if there is a max number of bins
            while max_bins<len(copyDF):  
                Chi2_as_num = pd.to_numeric(copyDF['Chi_2_if_merge'])
                index = Chi2_as_num.idxmin()
                copyDF = self.merge_2_bins_in_df(index-1,copyDF)
        self.chi2records = copyDF
        self.bin_intervals = copyDF.iloc[:,0:2].copy()
        return copyDF
    
    def fit_single_x(self, x_y_pair = None, method = 0, bin_rate = 0.01, critical_value=None, min_bins = 2, max_bins = None):
        if x_y_pair is None:
            x_y_pair = self.x_y_pair
            
        ## default method chi 2
        if method == 0:     
            self.init_equal_frequency(x_y_pair.drop(columns = [self.label]).iloc[:,0], bin_rate)
            self.mapping_bin(x_y_pair)
            self.calc_chi_2()
            self.chi2_merge_loop(critical_value=critical_value, min_bins = min_bins, max_bins = max_bins)
            
        ## write more methods here ---------------------------

    def transform_single_x(self, x_y_pair = None, bin_intervals = None):
        if x_y_pair is None:
            x_y_pair = self.x_y_pair
        return self.mapping_bin(x_y_pair, bin_intervals).bin
    
    def fit(self, original_df, label=None, cont_var_list = [], excepted_var_list =[], method = 0, bin_rate = 0.01, critical_value=None, min_bins = 2, max_bins = None):
        ## fitting entire given df, and store bin intervals in a df
        
        if label is None:
            label = self.label
        if len(cont_var_list) == 0:
            cont_var_list = original_df.columns.tolist()
            cont_var_list.remove(label)
            
        for col_name in excepted_var_list:
            try: 
                cont_var_list.remove(col_name)
            except:
                print(col_name," not in the cont_var_list")
            
        all_results = pd.DataFrame(columns = ["bin_intervals" , "fit_records"], index = cont_var_list)
        bin_intervals = []
        fit_records = []

        ## start to iterate through every variable
        for var in cont_var_list:                       
            xy_pair = original_df.loc[:,[var,label]]
            self.fit_single_x(xy_pair, method, bin_rate, critical_value, min_bins, max_bins)
            bin_intervals.append(self.bin_intervals)
            if method == 0:
                fit_records.append(self.chi2records)
        
        all_results.bin_intervals = bin_intervals
        all_results.fit_records = fit_records
        
        self.fit_results = all_results
        return all_results
    
    def transform(self, original_df, cont_var_list = [], excepted_var_list =[]):
        if len(cont_var_list) == 0:
            cont_var_list = self.fit_results.index.tolist()
            
        for col_name in excepted_var_list:
            try: 
                cont_var_list.remove(col_name)
            except:
                print(col_name," not in the cont_var_list")
                
        df = original_df.copy()
        
        for var in cont_var_list:
            df.loc[:,var] = self.transform_single_x(x_y_pair = original_df.loc[:,[var,self.label]], bin_intervals = self.fit_results.loc[var,'bin_intervals'])
        
        
        return df
            
        
    
        
