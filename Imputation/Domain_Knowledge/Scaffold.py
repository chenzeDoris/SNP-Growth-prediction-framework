
import CheckLG
import pandas as pd
from sklearn.impute import KNNImputer
from abc import ABC, abstractmethod

# test
class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def select_features(self,cluster, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DefaultFeatureSelection(FeatureSelectionStrategy):
    def select_features(self,cluster, data: pd.DataFrame) -> pd.DataFrame:
        # default feature selection : no feature selection
        return data.loc[:,cluster.Name]
class DM_FeatureSelection(FeatureSelectionStrategy):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def select_features(self,cluster,data: pd.DataFrame) -> pd.DataFrame:
        '''
        select features based on the allele frequency
        '''
        # caculate the allele frequency
        allele_fre_data = self.calculate_frequency(cluster,data)
        # Create a boolean condition for elements whose absolute difference is less than 0.05
        condition = abs(allele_fre_data.Allele_fre.diff()) <= self.threshold
        # When True retains the previous and current index. Then only keep one.
        rolling_condition = (condition | condition.shift(-1))
        filtered_data = allele_fre_data[rolling_condition]
        selected_features = []
        if len(filtered_data) != 0:
            Get_Highest_AF = allele_fre_data.iloc[filtered_data.Allele_fre.sort_values(
                ascending=False).index[0]].Name
            Get_lowest_AF = allele_fre_data.iloc[filtered_data.Allele_fre.sort_values(
            ).index[0]].Name
            # keep the highest allele frequency
            selected_features.append(Get_Highest_AF)
        # False means that no similarities were found and need to be retained.
        Get_False_AF = allele_fre_data[rolling_condition == False].Name
        if len(Get_False_AF) != 0:
            selected_features.extend(Get_False_AF)      
        
        return data.loc[:,selected_features]

    def calculate_frequency(self,cluster,snapperWithGeneName : pd.DataFrame) -> pd.DataFrame:
        '''
        caculate the allele frequency
        
        '''
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
            # if len(snapperWithoutNan) <= len(snapperWithGeneName) * 0.5:
            #     print('less common genes')
            for i in range(len(cluster)):
                # print(
                #     "Raw : ", snapperWithGeneName.iloc[:, i].value_counts().sum())
                i_sum = snapperWithoutNan.iloc[:, i].value_counts().sum()
                v_sum = snapperWithoutNan.iloc[:, i].sum()
                # print(snapperWithoutNan.iloc[:, i].value_counts())
                list_allele_fre.append([v_sum/(i_sum*2), snapperWithoutNan.iloc[:,
                                    i].name, cluster.iloc[i].Position, cluster.iloc[i].DBLabel])
        
        return pd.DataFrame(list_allele_fre, columns=['Allele_fre', 'Name', 'Postion', 'DBLabel'])        
        
class HandleScaffold:
    def __init__(self, snapperData, locationData):
        self.snapperData = snapperData
        self.locationData = locationData
        self.select_features_strategy = DefaultFeatureSelection()
        # imputer is the KNNImputer, features is the list of the featureName.
        self.record = {'Scaffold' : {'imputer':None,'features': None}}
        # self.snapper_without_missing = self.snapperData.iloc[:,2:].fillna(self.snapperData.mode().iloc[0])
    def set_feature_selection_strategy(self, strategy):
        '''
          # Use frequency-based feature selection
        handler.set_feature_selection_strategy(DM_FeatureSelection(threshold=0.05))
        '''
        if strategy:
            print('Custom Feature Selection')
            self.select_features_strategy = DM_FeatureSelection(threshold=0.05)  
        else:    
            print('Default Feature Selection')
            self.select_features_strategy = DefaultFeatureSelection()

             

    # def caculate_frequency(self, cluster, snapperWithGeneName):
    #     list_diff = cluster.Position.diff()
    #     list_allele_fre = []
    #     for i in range(len(cluster)):
    #         i_sum = snapperWithGeneName.iloc[:, i].value_counts().sum()
    #         v_sum = snapperWithGeneName.iloc[:, i].sum()
    #         list_allele_fre.append([v_sum/(i_sum*2), snapperWithGeneName.iloc[:,
    #                                i].name, cluster.iloc[i].Position, cluster.iloc[i].DBLabel])
    #     return list_allele_fre

    def check_allele_fre_withLabel(self, Data_withLabel, print=None):
        # fix bugs- selected features
        # DBLG = list_LG[0].run_DBSCAN(threshold)
        LG_geneName = Data_withLabel.groupby('DBLabel')
        list_Dataframe = []
        # index = []
        selected_data = pd.DataFrame()

        for label, cluster in LG_geneName:
            # Analysis
            GeneName = cluster.Name
            snapperWithGeneName = self.snapperData.loc[:, GeneName]
            seleted_cluster = self.select_features_strategy.select_features(cluster,snapperWithGeneName)
            selected_data = pd.concat([selected_data,seleted_cluster],axis=1)
            list_Dataframe.append(selected_data)
            # index.extend(cluster.Name)
            # if len(cluster) != 1:
            #     list_allele_fre = self.caculate_frequency(
            #         cluster, snapperWithGeneName)
            #     allele_fre_data = pd.DataFrame(list_allele_fre, columns=[
            #                                    'Allele_fre', 'Name', 'Postion', 'DBLabel'])
            #     # Create a boolean condition for elements whose absolute difference is less than 0.05
            #     condition = abs(allele_fre_data.Allele_fre.diff()) <= 0.05
            #     # When True retains the previous and current index. Then only keep one.
            #     rolling_condition = (condition | condition.shift(-1))
            #     filtered_data = allele_fre_data[rolling_condition]
            #     if len(filtered_data) != 0:

            #         Get_Highest_AF = allele_fre_data.iloc[filtered_data.Allele_fre.sort_values(
            #             ascending=False).index[0]].Name
            #         Get_lowest_AF = allele_fre_data.iloc[filtered_data.Allele_fre.sort_values(
            #         ).index[0]].Name
            #         index.append(Get_Highest_AF)
            #     # False means that no similarities were found and need to be retained.
            #     Get_False_AF = allele_fre_data[rolling_condition == False].Name

            #     if len(Get_False_AF) != 0:
            #         index.extend(Get_False_AF)
            #     list_Dataframe.append(allele_fre_data)
            # else:
            #     index.extend(cluster.Name)

        return list_Dataframe, selected_data

    '''Return to a dataframe with no Missing values.'''

    def get_Scaffold_Name(self, threshold, Scaffold_data, snapper):
        list_data = []
        # get all the Scaffold Name
        for j in Scaffold_data.index:
            Scaffold = CheckLG.checkLG(snapper, self.locationData, j)
            DBscaffold = Scaffold.run_DBSCAN(threshold)
            # print("----------------------------------------{} : {}--------------------------------------".format(Scaffold.str,threshold))
            # print("Number of Features : ",len(DBscaffold))
            # print("Number of clusters : ",len(DBscaffold.DBLabel.value_counts()))
            # print("Reduce :" ,len(DBscaffold)-len(DBscaffold.DBLabel.value_counts()))
            DD_sca, selected_data_sca = self.check_allele_fre_withLabel(
                DBscaffold)
            list_data.extend(selected_data_sca)
            # get a dataframe
            selected_sca = snapper.loc[:, list_data]

        return selected_sca

    def Use_KNN(self, data):
        # Use KNN to impute
        imputer = KNNImputer(n_neighbors=2)
        data_imputed = imputer.fit_transform(data)
        data_imputed = data_imputed.round().astype(int)
        data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
        # data = data_imputed_df
        return data_imputed_df, imputer
    # this merge all the scaffold and get selected genes. Main

    def handle_Scaffold(self, location_scaffold, threshold, snapper):
        # make all uppercase
        location_scaffold['Chromosome/scaffold'] = location_scaffold['Chromosome/scaffold'].str.upper()
        # make all the symbol same
        location_scaffold['Chromosome/scaffold'] = location_scaffold['Chromosome/scaffold'].str.replace(
            '_', '.')
        # get ALL scaffold.
        new_vlauecount = pd.DataFrame(
            location_scaffold['Chromosome/scaffold'].value_counts())
        # this have problem
        
        print('Scaffold number : ', new_vlauecount)
        # check if they added up same witht eh snapper shape 
        print("sum of the SNPs",new_vlauecount.sum())
        # only keep scaffolds
        # Get data from index 25 onwards (25 IS included) and filter for count > 1
        Scaffold_data = new_vlauecount[25:]
        Scaffold_data = Scaffold_data[Scaffold_data['count'] > 1]
        print('Scaffold_data', Scaffold_data)
        # Scaffold_data = new_vlauecount[(
        #     new_vlauecount['count'] < 10) & (new_vlauecount['count'] > 1)]
        # if only one genes in Scaffold.
        Scaffold_data_1 = new_vlauecount[new_vlauecount['count'] == 1]
        Scaffold_1_Name = [
            name for i in Scaffold_data_1.index for name in self.locationData[self.locationData['Chromosome/scaffold'].str.contains(i)].Name.values]
        Scaffold_1_Data = snapper.loc[:, snapper.columns.isin(Scaffold_1_Name)]
        
        # get all selected Scaffold
        selected_sca_Imp = self.get_Scaffold_Name(
            threshold, Scaffold_data, snapper)
        # concatenate
        selected_sca_total = pd.concat(
            [selected_sca_Imp, Scaffold_1_Data], axis=1)
        print('Scaffold total shape : ', selected_sca_total.shape)
        # handle missing values
        selected_sca_final, imputer = self.Use_KNN(selected_sca_total)
        self.record['Scaffold']['imputer'] = imputer
        self.record['Scaffold']['features'] = selected_sca_final.columns
        # check missing values
        if selected_sca_final.isnull().sum().sum() > 0:
            print('Scaffold still have missing values')
            print(selected_sca_final.isnull().sum())
        #TODO 
        # sys.exit(0)  # Exit the program if needed, remove this line in production
        return selected_sca_final
