
import pandas as pd
import numpy as np


class HandleMissingData():
    def __init__(self,list_LG_15, snapper):
        '''
        :location: location dataset to get chromosomes and locations
        :data: data set to help find the gene values
        :str: the name of the chromosome sequence we need
        '''
        self.snapper = snapper
        self.list_LG_15 = list_LG_15
        self.record = {}
        self.LGname = None
        
    

    def caculate_frequency_fix(self,cluster, snapperWithGeneName):
        list_diff = cluster.Position.diff()
        list_allele_fre = []
        less_missing_Name = ''
        if len(cluster) == 1:
            i_sum = snapperWithGeneName.iloc[:, 0].value_counts().sum()
            v_sum = snapperWithGeneName.iloc[:, 0].sum()
            list_allele_fre.append([v_sum/(i_sum*2), snapperWithGeneName.iloc[:,
                                0].name, cluster.iloc[0].Position, cluster.iloc[0].DBLabel])
        else:
            snapperWithoutNan = snapperWithGeneName.dropna()
            if len(snapperWithoutNan) <= len(snapperWithGeneName) * 0.5:
                print('less common genes')
            for i in range(len(cluster)):
                i_sum = snapperWithoutNan.iloc[:, i].value_counts().sum()
                v_sum = snapperWithoutNan.iloc[:, i].sum()
                list_allele_fre.append([v_sum/(i_sum*2), snapperWithoutNan.iloc[:,
                                    i].name, cluster.iloc[i].Position, cluster.iloc[i].DBLabel])
        return list_allele_fre


  


    def check_diversity(self,snapperName):
        '''
        For remove one feature, I perfer to use allele frequency as the  main rule
        Rule for keep the gene : 
        1. keep the one more diversit.
        2. keep the one less missing value

        snapperName : Cluster genes in snapper data
        return : Name of the gene
        '''

        list_name = snapperName.columns.tolist()
        #  keep the one more diversit
        unique_counts = snapperName.nunique()
        is_most_diverse_unique = (unique_counts == unique_counts.max()).sum() == 1
        if is_most_diverse_unique:
            # Find the column with the most diverse values
            most_diverse_column = unique_counts.idxmax()
            print('more diversity :', most_diverse_column)
            # Got other names to fill in.
            other_value = list_name[1 - list_name.index(most_diverse_column)]
            return most_diverse_column, other_value
        else:
            # if both same, then use less missing values
            less_missing = snapperName.isnull().mean().idxmin()
            print("less miss _values :", snapperName.isnull().mean().idxmin())
            other_value = list_name[1 - list_name.index(less_missing)]
            return less_missing, other_value


    '''
    snapperName : Cluster genes in snapper data
    correlation_threshold : Threshold for coefficient values.
    otherInfo : include Feature Name, Position and allele frequency
    '''


    def make_columns_pair(self,corr_mask):
        corr_columns = []
        for col1, col2 in zip(*np.where(corr_mask)):
            col1_name = corr_mask.columns[col1]
            col2_name = corr_mask.columns[col2]
            # Sort the column names to avoid duplicates
            pair = tuple(sorted([col1_name, col2_name]))
            if pair not in corr_columns:
                corr_columns.append(pair)

        return corr_columns


    def check_correlation(self,snapperName, correlation_threshold, otherInfo, LGstr):
        list_negative_corr = []
        # check the correlation for the cluster genes
        corr_matrix = snapperName.corr()
        # coro_matrix_kendall = snapperName.corr(method='kendall')
        # positive
        high_corr_mask = (corr_matrix >= correlation_threshold) & (
            corr_matrix != 1)
        low_corr_mask = (
            corr_matrix <= -correlation_threshold) & (corr_matrix != 1)
        # Keep only the high correlation values
        high_corr_df = corr_matrix[high_corr_mask]
        # fin high_corr
        high_corr_columns = self.make_columns_pair(high_corr_mask)

        # find low_corr
        negative_corr_pairs = []
        for col1, col2 in zip(*np.where(low_corr_mask)):
            col1_name = low_corr_mask.columns[col1]
            col2_name = low_corr_mask.columns[col2]
            # Sort the column names to avoid duplicates
            pair = tuple(sorted([col1_name, col2_name]))

            if pair not in negative_corr_pairs:
                negative_corr_pairs.append(pair)
        # This is used to export all the negative corr for Julie
        for pair in negative_corr_pairs:
            col1, col2 = pair
            correlation_coefficient = corr_matrix.at[col1, col2]
            # print(
            #     f"Correlation coefficient between '{col1}' and '{col2}': {correlation_coefficient}")
            col1_info = otherInfo[otherInfo.Name == col1]
            col1Position = col1_info.Position.iloc[0]
            col1Alle = col1_info['allele frequency'].iloc[0]
            col2_info = otherInfo[otherInfo.Name == col2]
            col2Position = col2_info.Position.iloc[0]
            col2Alle = col2_info['allele frequency'].iloc[0]
            list_negative_corr.append(
                [col1, col1Position, col1Alle, col2, col2Position, col2Alle, correlation_coefficient, LGstr])
        # print("The info, : ", list_negative_corr)
        # print("high_corr: ", high_corr_columns)
        # print("low_corr :", negative_corr_pairs)
        # get the name of the columns
        high_corr_list = [
            col for col in high_corr_mask.columns if any(high_corr_mask[col])]
        low_corr_list = [
            col for col in low_corr_mask.columns if any(low_corr_mask[col])]

        return high_corr_columns, negative_corr_pairs, list_negative_corr, high_corr_list, low_corr_list


    '''
    snapperData : raw data
    corr_columns : high or low correlation columns 

    Return (Dataframe): Filled feature 
    '''


    def handle_missing_values_Domain_high(self,snapperData, corr_columns, high_name_list):
        raw_cluster = snapperData.loc[:, high_name_list]
        
        for names in corr_columns:
            # get cluster
            cluster = raw_cluster.loc[:, names]
            find_nan = cluster.isna()
            index = find_nan.index[find_nan.any(axis=1)]
            mapping_dict_AtoB = self.run_test_filled(cluster, order='AB')
            mapping_dict_BtoA = self.run_test_filled(cluster, order='BA')
            
            A = names[0]
            B = names[1]

            # print(A, 'and', B)
            # {A: B, dictAtoB : mapping_dict_AtoB, B: A, dictBtoA : mapping_dict_BtoA}
            self.record[self.LGname].append({A: B, 'AtoB' : mapping_dict_AtoB, B: A, 'BtoA' : mapping_dict_BtoA})

            # Apply the mapping to fill missing values in column A based on column B
            raw_cluster.loc[:, A] = raw_cluster.loc[:, A].fillna(
                raw_cluster.loc[:, B].map(mapping_dict_BtoA))
            raw_cluster.loc[:, B] = raw_cluster.loc[:, B].fillna(
                raw_cluster.loc[:, A].map(mapping_dict_AtoB))

        return raw_cluster


    def run_test_filled(self,cluster, order='AB'):
        '''
        cluster : cluster data
        order : AB or BA
        '''
        # Get the unique combinations of values in the cluster
        combination_counts = []
        if order == 'AB':
            for index, count in cluster.value_counts().items():
                combination_counts.append({index[0]: index[1], 'count': count})
        elif order == 'BA':
            for index, count in cluster.value_counts().items():
                combination_counts.append({index[1]: index[0], 'count': count})

        # Create a dictionary to store counts for each unique pair
        pair_counts = {}

        # Iterate through the data and update counts
        for item in combination_counts:
            keys = list(item.keys())
            key1 = keys[0]  # Get the first key
            key2 = item[key1]  # Get the corresponding value
            count = item['count']
            pair_key = (key1, key2)
            if pair_key[0] in [k[0] for k in pair_counts.keys()]:
                existing_key = next(k for k in pair_counts.keys()
                                    if k[0] == pair_key[0])
                if count > pair_counts[existing_key]:
                    pair_counts[existing_key] = count
            else:
                pair_counts[pair_key] = count

        # Convert the pair counts back to the desired format
        result = {pair_key[0]: pair_key[1]
                for pair_key, count in pair_counts.items()}

        return result


    def handle_missing_values_Domain_low(self,snapperData, low_corr_columns, low_name_list):
        raw_cluster = snapperData.loc[:, low_name_list]
        
        for names in low_corr_columns:
            # print("-----------------started to filled --------------------------")
            # get cluster
            cluster = raw_cluster.loc[:, names]
            # find Nan
            find_nan = cluster.isna()
            index = find_nan.index[find_nan.any(axis=1)]
            # # check diversity
            # SNP_diversity, sample = check_diversity(cluster)
            # AB means index 0:1,  BA means index 1 : 0
            mapping_dict_AtoB = self.run_test_filled(cluster, order='AB')
            mapping_dict_BtoA = self.run_test_filled(cluster, order='BA')
            A = names[0]
            B = names[1]
            self.record[self.LGname].append({A: B, 'AtoB' : mapping_dict_AtoB, B: A, 'BtoA' : mapping_dict_BtoA})
            # print(A, 'and', B)
            # Apply the mapping to fill missing values in column A based on column B
            raw_cluster.loc[:, A] = raw_cluster.loc[:, A].fillna(
                raw_cluster.loc[:, B].map(mapping_dict_BtoA))
            raw_cluster.loc[:, B] = raw_cluster.loc[:, B].fillna(
                raw_cluster.loc[:, A].map(mapping_dict_AtoB))
            # print("-----------------end to filled --------------------------")
        return raw_cluster


    def run_code_HMD(self,threshold=10000):
        '''
        list_LG_15 : list of LG
        old_snapper : raw data
        '''
        
        list_LG_15 = self.list_LG_15
        old_snapper = self.snapper
        Handle_missing_Data = pd.DataFrame()

        negative_dataframe = pd.DataFrame(columns=['Feature ID_1', 'Position 1', 'allele frequency 1',
                                        'Feature ID_2', 'Position 2', 'allele frequency 2', 'correlation coefficient', 'LG'])
        for LG in list_LG_15:
            # print("====================={}================".format(LG.str))
            # get the cluster, the threshold is 10000
            labelstemp = LG.run_DBSCAN(threshold)
            con = 0
            self.record[LG.str] = []
            self.LGname =LG.str
            for i, c in labelstemp.groupby('DBLabel'):
                list_allele_fre = self.caculate_frequency_fix(
                    c, old_snapper.loc[:, c.Name])

                if len(c) > 1:
                    # caculate the allele frequency
                    convert_pd = pd.DataFrame(list_allele_fre, columns=[
                        'allele frequency', 'Name', 'Position', 'labels'])
                    # print(convert_pd)
                    snapperName = old_snapper.loc[:, convert_pd.Name]
                    # check how many Nan
                    find_nan = old_snapper.loc[:, convert_pd.Name].isna()
                    index = find_nan.index[find_nan.any(axis=1)]
                    # print(old_snapper.loc[:, convert_pd.Name])
                    # here
                    high_corr_columns, low_corr_columns, negative_list, high_list, low_list = self.check_correlation(
                        snapperName, 0.6, convert_pd, LG.str)
                    if len(negative_list) > 0:

                        negative_list_pd = pd.DataFrame(negative_list, columns=[
                                                        'Feature ID_1', 'Position 1', 'allele frequency 1', 'Feature ID_2', 'Position 2', 'allele frequency 2', 'correlation coefficient', 'LG'])
                        negative_dataframe = pd.concat(
                            [negative_list_pd, negative_dataframe], ignore_index=True)

                    if len(high_corr_columns) > 0:
                        filled_column = self.handle_missing_values_Domain_high(
                            old_snapper, high_corr_columns, high_list)
                        # print("test_high : ", filled_column)
                        Handle_missing_Data = pd.concat(
                            [filled_column, Handle_missing_Data], axis=1)
                        con += 1
                    if len(low_corr_columns) > 0:
                        filled_column = self.handle_missing_values_Domain_low(
                            old_snapper, low_corr_columns, low_list)
                        # print("test_low : ", filled_column.isna().sum())
                        Handle_missing_Data = pd.concat(
                            [filled_column, Handle_missing_Data], axis=1)
                        # Fill NaN values in column 'A' with corresponding values from column 'B'
                        con += 1

            Handle_missing_Data = Handle_missing_Data.loc[:,
                                                        ~Handle_missing_Data.columns.duplicated()]
            # print(Handle_missing_Data)
        # negative_dataframe.to_csv('negatively_correlated_genes_0.7.csv')
        return Handle_missing_Data


