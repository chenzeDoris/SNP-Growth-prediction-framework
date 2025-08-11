#%%
import time
import argparse
# from Scaffold import HandleScaffold
from CheckLG import checkLG
import pandas as pd
from run_allele_frequency import run_sss_with_total
from Handle_Missing_Dom import HandleMissingData
from sklearn.impute import KNNImputer

# nohup python -u Main.py >filled_0.6_test.out 2>&1 &
'''
all the correlation and threshold is 0.6
the allele frequency < 0.05
'''
# snapper = pd.read_csv('../data_clf_MV50.csv', index_col=0)

# location = pd.read_excel('../Snapper_SNP_locations.xlsx')

class Run_DMimputation() :
    def __init__(self, snapper, location,FS = False):
        self.snapper = snapper
        self.location = location
        self.FS = FS
        self.record = {'LG': [], 'Scaffold': []}
    def transform(self,Data):
        # check self.record is None or not
        if self.record is None:
            print('No record')
            return Data
        def Use_KNN(data,imputer):
            if isinstance(data, pd.Series):
                data = data.to_frame()
            # Use KNN to impute
            data_imputed = imputer.transform(data)
            data_imputed = data_imputed.round().astype(int)
            data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns,index=data.index)
            # check missing value 
            # if data_imputed_df.isnull().sum().sum() > 0:
            #     # print('KNN-Still have missing values')
            #     print(data_imputed_df.isnull().sum())
            # else:
            #     print('KNN-All missing values have been imputed')    
            # data = data_imputed_df
            return data_imputed_df
        # DO I really need to record the KNN imputer?
        # based on the record to transform the data
        def fill_missing_values(data, A, B, mapping_dict_AtoB, mapping_dict_BtoA):
            # Convert mapping dictionaries to pandas Series for efficient mapping
            map_series_AtoB = pd.Series(mapping_dict_AtoB)
            map_series_BtoA = pd.Series(mapping_dict_BtoA)
            # Fill missing values in A using B
            mask_A = data[A].isnull()
            data.loc[mask_A, A] = data.loc[mask_A, B].map(map_series_BtoA)
            # Fill missing values in B using A
            mask_B = data[B].isnull()
            data.loc[mask_B, B] = data.loc[mask_B, A].map(map_series_AtoB)
            return data       
        # fit to the DM imputer
        for name in list(self.record['LG'].keys()):
            print('LG',len(self.record['LG'][name]))
            for i in range(len(self.record['LG'][name])):
                pairs = self.record['LG'][name][i]
                key_list = list(self.record['LG'][name][i].keys())
                A = pairs[key_list[0]]
                B = pairs[key_list[2]]
                mapping_dict_AtoB = pairs[key_list[1]]
                mapping_dict_BtoA = pairs[key_list[3]]
                # print(mapping_dict_AtoB)
                Data = fill_missing_values(Data,A,B,mapping_dict_AtoB,mapping_dict_BtoA)

         # using KNN on LG
        TotalLGSed = pd.DataFrame() 
        for i in range(len(self.record['KNN'])):
            KNN = self.record['KNN'][i]
            LG = list(KNN.keys())[0]
            # print('LG',LG)
            LG_selectd_feature = list(KNN[LG]['selected_data_LG'])
            KNNimputer = KNN[LG]['KNN']
            # based on the selected feature to do the imputation
            LGData = Use_KNN(Data.loc[:,LG_selectd_feature],KNNimputer)
            TotalLGSed = pd.concat([TotalLGSed,LGData], axis=1)
        # do on scaffold
        Scaffold = self.record['Scaffold']['Scaffold']
        knn_imputer = Scaffold['imputer']
        selected_feature = list(Scaffold['features'])
        scaffold = Use_KNN(Data.loc[:,selected_feature],knn_imputer)

        # print('after imputation',Data)
        Total_selected_data = pd.concat([TotalLGSed, scaffold], axis=1)
        Total_selected_data = Total_selected_data.loc[:, ~Total_selected_data.columns.duplicated()]
        # print('Total_selected_data',Total_selected_data)
        # check the missing values
        # print('after imputation')
        # print(Total_selected_data.isnull().sum())

        return Total_selected_data
        
        
    def _getrecord(self):
        return self.record   
    def _getFs(self):
        return self.FS 
    
    def run(self,split = False,threshold = 10000):
        '''
        split : True means the input data is X-train, X_test. don't need to concat.
        split : False means imputation for whole data.
        '''
        # check the location shape
        print('Location shape:', self.location.shape)
        # # make all the letter in the Chromosome/scaffold to be upper case
        self.location['Chromosome/scaffold'] = self.location['Chromosome/scaffold'].str.upper()
        location_values = pd.DataFrame(self.location['Chromosome/scaffold'].value_counts())
        location_rank = location_values[location_values['count'] > 5].index
        # This only keep 24
        location_rank = location_rank[:25]
        # location_rank = location_rank.delete(25)
        #
        list_LG_15 = []
        for i in location_rank:
            LG = checkLG(self.snapper, self.location, i)
            ind = LG.LG.Name.isin(LG.snapperLG.columns)
            LG.LG = LG.LG.loc[ind]
            list_LG_15.append(LG)
        hmd = HandleMissingData(list_LG_15, self.snapper)
        
        print("Threshold for DM imputation:", threshold)
        handle_missing_Data = hmd.run_code_HMD(threshold=threshold)
        self.record['LG'] = hmd.record
        # handle_missing_Data = run_code_HMD(list_LG_15, self.snapper)
        columns_to_merge = self.snapper.columns.difference(handle_missing_Data.columns)
        merged_df = pd.concat([self.snapper[columns_to_merge[::-1]],
                            handle_missing_Data], axis=1)
        print("LG after imputed",merged_df.shape)
        if merged_df.isnull().sum().sum() > 0:
            print('still have missing values')
            print(merged_df.isnull().sum())
        #  this is do the feture seletion
        list_LG_50 = []
        for i in location_rank:
            LG = checkLG(merged_df, self.location, i)
            ind = LG.LG.Name.isin(LG.snapperLG.columns)
            LG.LG = LG.LG.loc[ind]
            list_LG_50.append(LG)

        final_selected,KNN_inform = run_sss_with_total(
            snapper=merged_df, list_LG=list_LG_50, location=self.location, threshold=threshold,record=self.record,FS = self.FS)
        # this added the other labels
        # for train and test don't need to concat.
        self.record['KNN'] = KNN_inform
        if split:
            print('Data is split, no need to concat')
            return final_selected
        else:
            # this Growth_rate , Parents ,Class , medium_pro
            final_data = pd.concat([self.snapper.iloc[:, :4],
                                final_selected], axis=1)
            
            return final_data



#%%

# Run_DMimputation(snapper, location,False).run().to_csv('Redata_without_DMFS.csv')
# Run_DMimputation(snapper, location,True).run().to_csv('Redata_with_DMFS.csv')

# %%
# print(snapper)
# def remove_duplicate(data):
#     # Find columns where all values are the same
#     columns_to_remove = data.columns[data.nunique() == 1]
#     # Remove those columns from the DataFrame
#     df_filtered = data.drop(columns=columns_to_remove)
#     return df_filtered

# print("Do the feature selection")
# test_data = Run_withoutFS(snapper, location,True).run()
# print(test_data)
# # remove the duplicate columns
# test_data = remove_duplicate(test_data)

# # export to the dataset
# test_data.to_csv('Redata_with_DMFS.csv')

# %%

# got the all the chromosmes/scaffold count

# location_values = pd.DataFrame(location['Chromosome/scaffold'].value_counts())
# print(location_values)
# # got the all the chromosmes/scaffold count which is greater than 5
# location_rank = location_values[location_values['count'] > 5].index
# print(location_rank)
# # deleted the 25th index which is the Scaffold, not chromosome
# location_rank = location_rank.delete(25)
# print(snapper.iloc[:, :3])


# # this for the chromosome
# list_LG_15 = []
# for i in location_rank:
#     # get the LG
#     LG = checkLG(snapper, location, i)
#     # update the LG
#     ind = LG.LG.Name.isin(LG.snapperLG.columns)
#     LG.LG = LG.LG.loc[ind]
#     list_LG_15.append(LG)

# handle_missing_Data = run_code_HMD(list_LG_15, snapper)
# print("still have Nan:", handle_missing_Data.isnull().sum())

# # Identify columns in df1 that are not in df2
# columns_to_merge = snapper.columns.difference(handle_missing_Data.columns)
# # Create a new DataFrame by concatenating selected columns
# merged_df = pd.concat([snapper[columns_to_merge[::-1]],
#                       handle_missing_Data], axis=1)

# # check whole data have missing or not after imputation
# # if still have the missing value, return the sum of missing values
# if merged_df.isnull().sum().sum() > 0:
#     print('still have missing values')
#     print(merged_df.isnull().sum())


# list_LG_50 = []
# for i in location_rank:
#     LG = checkLG(merged_df, location, i)
#     # update the LG
#     ind = LG.LG.Name.isin(LG.snapperLG.columns)
#     LG.LG = LG.LG.loc[ind]
#     list_LG_50.append(LG)

# final_selected = run_sss_with_total(
#     snapper=merged_df, list_LG=list_LG_50, location=location, threshold=10000)

# final_data = pd.concat([snapper.iloc[:, :2],
#                         final_selected], axis=1)
# print(final_data)

# export to the dataset
# This data did not have the ID and need to relabelled.
# final_data.to_csv('Redata_without_FS.csv')