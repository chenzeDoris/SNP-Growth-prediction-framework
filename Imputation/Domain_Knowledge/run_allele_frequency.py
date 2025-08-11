import pandas as pd
import numpy as np
from Scaffold import HandleScaffold
from sklearn.impute import KNNImputer

# all the print. 
def get_missing_rate(data, percentage):
    '''
    data : Raw data
    percentage: Percentage of missing rates.
    '''
    missing_data_in_df = pd.DataFrame(
        {'NaN_Counts': data.isna().sum(), 'NaN_Proportions(%)': (data.isna().sum() / data.shape[0]) * 100}).sort_values(
        by='NaN_Counts', ascending=False)
    missing_data_po = missing_data_in_df.loc[
        missing_data_in_df["NaN_Proportions(%)"] > percentage]
    # delete missing value that exceed a percentage
    data_without_miss = data.drop(columns=missing_data_po.index.values)

    return data_without_miss


# double-check
def caculate_frequency(cluster, snapperWithGeneName):
    '''
    cluster: cluster
    snapperWithGeneName : snapper data.
    '''
    # find the common instances
    list_diff = cluster.Position.diff()
    list_allele_fre = []
    if len(cluster) == 1:
        for i in range(len(cluster)):
            i_sum = snapperWithGeneName.iloc[:, i].value_counts().sum()
            v_sum = snapperWithGeneName.iloc[:, i].sum()
            list_allele_fre.append([v_sum/(i_sum*2), snapperWithGeneName.iloc[:,
                                   i].name, cluster.iloc[i].Position, cluster.iloc[i].DBLabel])
    else:
        snapperWithoutNan = snapperWithGeneName.dropna()
        if len(snapperWithoutNan) < len(snapperWithGeneName) * 0.5:
            print('less common genes')
        for i in range(len(cluster)):
            i_sum = snapperWithoutNan.iloc[:, i].value_counts().sum()
            v_sum = snapperWithoutNan.iloc[:, i].sum()
            list_allele_fre.append([v_sum/(i_sum*2), snapperWithoutNan.iloc[:,
                                   i].name, cluster.iloc[i].Position, cluster.iloc[i].DBLabel])
    return list_allele_fre


def check_diversity(snapperName):
    '''
For remove one feature, I perfer to use allele frequency as the  main rule
Rule for keep the gene : 
1. keep the one more diversit.
2.keep the one less missing value

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
        # print('more diversity :', most_diverse_column)
        # Got other names to fill in.
        other_value = list_name[1 - list_name.index(most_diverse_column)]
        return most_diverse_column, other_value
    else:
        # if both same, then use less missing values
        less_missing = snapperName.isnull().mean().idxmin()
        # print("less miss _values :", snapperName.isnull().mean().idxmin())
        other_value = list_name[1 - list_name.index(less_missing)]
        return less_missing, other_value


def check_negative_correlation(snapperName, correlation_threshold):
    # check the correlation for the cluster genes
    corr_matrix = snapperName.corr()
    # print(corr_matrix)
    # negative_correlation
    low_corr_mask = (
        corr_matrix <= -correlation_threshold) & (corr_matrix != 1)
    # find low_corr
    negative_corr_pairs = []
    for col1, col2 in zip(*np.where(low_corr_mask)):
        col1_name = low_corr_mask.columns[col1]
        col2_name = low_corr_mask.columns[col2]
        # Sort the column names to avoid duplicates
        pair = tuple(sorted([col1_name, col2_name]))
        if pair not in negative_corr_pairs:
            negative_corr_pairs.append(pair)
    # print("low_corr :", negative_corr_pairs)
    return negative_corr_pairs


def new_check_allele_fre_withLabel(Data_withLabel, snapper, FS=False):
    '''
    Data_withLabel : Data with label
    snapper : snapper data
    print : print the result
    This function is used to check the allele frequency for the cluster genes. 
    Then check the negative correlation.
    Then only keep the one with the highest allele frequency(more diversity)
    The allele diff lower than 0.05 will be consider to remove one of them.
    '''
    # fix bugs- selected features
    # DBLG = list_LG[0].run_DBSCAN(threshold)
    
    LG_geneName = Data_withLabel.groupby('DBLabel')
    list_Dataframe = []
    index = []

    for label, cluster in LG_geneName:
        # Analysis
        GeneName = cluster.Name

        snapperWithGeneName = snapper.loc[:, GeneName]
        
        if FS:
            # if the cluster size is 1, then keep the gene
            if len(cluster) != 1:
                list_allele_fre = caculate_frequency(cluster, snapperWithGeneName)
                list_negative_cor = check_negative_correlation(
                    snapperWithGeneName, 0.6)
                allele_fre_data = pd.DataFrame(list_allele_fre, columns=[
                                               'Allele_fre', 'Name', 'Postion', 'DBLabel'])
                # Create a boolean condition for elements whose absolute difference is less than 0.05
                condition = abs(allele_fre_data.Allele_fre.diff()) <= 0.05
                # When True retains the previous and current index. Then only keep one.
                rolling_condition = (condition | condition.shift(-1))
                filtered_data = allele_fre_data[rolling_condition]
                if len(filtered_data) != 0:
                    SNP, othertest = check_diversity(
                        snapper.loc[:, filtered_data.Name])
                    # Get_Highest_AF = allele_fre_data.iloc[filtered_data.Allele_fre.sort_values(ascending=False).index[0]].Name
                    # Get_lowest_AF = allele_fre_datat.iloc[filtered_data.Allele_fre.sort_values().index[0]].Name
                    index.append(SNP)

                if len(list_negative_cor) > 0:
                    temp_names = []
                    for genes in list_negative_cor:
                        SNP, othertest = check_diversity(snapper.loc[:, genes])
                        temp_names.append(SNP)
                    temp_names = list(set(temp_names))
                    index += temp_names
                # False means that no similarities were found and need to be retained.
                Get_False_AF = allele_fre_data[rolling_condition == False].Name
                if len(Get_False_AF) != 0:
                    # check False values have no overlap with negative correlation
                    has_intersection = any(set(inner) & set(Get_False_AF)
                                           for inner in list_negative_cor)
                    if not has_intersection:
                        index.extend(Get_False_AF)
                list_Dataframe.append(allele_fre_data)
            else:
                index.extend(cluster.Name)
        else:
            # keep all the genes
            index.extend(cluster.Name)
    
        
    return list_Dataframe, snapper.loc[:, list(set(index))], list(set(index))



def Use_KNN(data):

    # Use KNN to impute
    imputer = KNNImputer(n_neighbors=2)
    data_imputed = imputer.fit_transform(data)
    data_imputed = data_imputed.round().astype(int)
    data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns,index=data.index)
    # check missing value 
    if data_imputed_df.isnull().sum().sum() > 0:
        print('KNN-Still have missing values')
        print(data_imputed_df.isnull().sum())
    # else:
        # print('KNN-All missing values have been imputed')    
    # data = data_imputed_df
    return data_imputed_df,imputer


def run_LG(snapper, list_LG, threshold, FS=False):
    list_withThre_selected = []
    Total = 0
    Total_LG = 0
    TotalLGSed = pd.DataFrame()
    KNN_imputed = []
    for LG in list_LG:
        Total_LG = len(LG.get_the_LG().columns)+Total_LG
        # print("----------------------------------------{} : {}--------------------------------------".format(LG.str, threshold))
    # print("position threshold : {}".format(threshold))
        DBLG = LG.run_DBSCAN(threshold)
        # print(DBLG)
        # DD,selected_data = check_allele_frequency_withLabel(DBLG,snapper)
        DD, selected_data, test_va = new_check_allele_fre_withLabel(
            DBLG, snapper,FS)
        # DD refers to the gene information of all gene blocks. Where the cluster size >1
        Total = len(DD)+Total
        # use KNN imputation missing values
        selected_data_LG,imputer = Use_KNN(selected_data)
        KNN_imputed.append({LG.str:{'KNN' :imputer, 'selected_data_LG': selected_data_LG.columns.tolist()}})
        # print(selected_data_LG)
        TotalLGSed = pd.concat([TotalLGSed, selected_data_LG], axis=1)

    return TotalLGSed,KNN_imputed


def run_sss_with_total(snapper, list_LG, location, threshold,record = None,FS=True):
    '''
    snapper : snapper data
    list_LG : list of LG objects
    location : location data
    threshold : threshold for DBSCAN
    record : record for the imputation
    FS : feature selection strategy (DKFS)
    '''
    print("threshold : ", threshold)
    Scaffold = HandleScaffold(snapperData=snapper, locationData=location)
    Scaffold.set_feature_selection_strategy(FS)
    scaffold_data = Scaffold.handle_Scaffold(location, threshold, snapper)
    print("scaffold_data : ", scaffold_data)
    LG,KNN_inform = run_LG(snapper, list_LG, threshold,FS)
    print("LG : ", LG)
    Total_selected_data = pd.concat([LG, scaffold_data], axis=1)
    Total_selected_data = Total_selected_data.loc[:,
                                                  ~Total_selected_data.columns.duplicated()]
    print("Total_selected data ; ",Total_selected_data)
    if Total_selected_data.isnull().sum().sum() > 0:
        print('Total still have missing values')
        print(Total_selected_data.iloc[:,3:].isnull().sum())
    else:
        print('All missing values have been imputed')  

    record['Scaffold'] = Scaffold.record
    return Total_selected_data,KNN_inform
