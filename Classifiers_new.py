#Classification
#%%
from sklearn.svm import SVC
from dwd.socp_dwd import DWD
# from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef
# logger
from loguru import logger
# Kflod and Split
import pandas as pd
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import pickle
import psutil
from typing import Dict, Any, Union
# fairness
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio
# import the feature selection method
from FeatureSelection_new import FeatureSelection
# import the imputation methods
import sys
import os
#  path from the domain knowledge

sys.path.append('./Imputation/Domain_Knowledge')
from Run_theRedata_witoutFS import Run_DMimputation

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        print('Accuracy:',acc)
        return float(f"{acc:.3f}")
    @staticmethod
    def baseline(y_true,y_pred):
        # calculate the baseline
        # the baseline is the max of the Fast and Slow
        Fast_baseline = len(y_true.loc[y_true == 'FAST'])/len(y_true)
        Slow_baseline = len(y_true.loc[y_true == 'SLOW'])/len(y_true)
        return Fast_baseline,Slow_baseline
    @staticmethod
    def AUC_ROC(y_true, y_pred_proba):
        """Calculate ROC AUC score
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities from classifier.predict_proba()
            
        Returns:
            float: ROC AUC score
        """
        try:

            # Convert string labels to numeric if needed
            if isinstance(y_true[0], str):
                y_true = label_binarize(y_true, classes=['SLOW', 'FAST'])
            
            # For binary classification, use the probability of the positive class
            if y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            print('roc_auc:',roc_auc)
            return float(f"{roc_auc:.3f}")
            
        except Exception as e:
            print(f"Error calculating AUC-ROC: {str(e)}")
            return None
    @staticmethod
    def classification_report(y_true, y_pred):
        return classification_report(y_true, y_pred)
    @staticmethod
    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
    
    
    @staticmethod
    def misclassified_indices(y_true, y_pred):
        y_true_series = pd.Series(y_true)
        y_pred_series = pd.Series(y_pred, index=y_true_series.index)
        return y_true_series.index[y_true_series != y_pred_series]
    
    @staticmethod
    def stat_test_Man(acclist1,acclist2):

        stat, p = mannwhitneyu(acclist1, acclist2)
        if p<0.05:
            print(f"With signification differencce : U = {stat}, p-value = {p}")
        else: print(f'{pair} No significant difference')

        
        return y_true_series.index[y_true_series != y_pred_series]
    @staticmethod
    def population_accuracy(y_true, y_pred,data= None):
        '''
        data: data is the index of the y_test or y_train
        '''
        if data is None:
            return 0
        if 'Parents' not in data.columns:
            print("Data does not have 'Parents' column, cannot calculate population accuracy.")
            return 0
        missclass = Metrics.misclassified_indices(y_true, y_pred)
        # print('missclass',missclass)
        dic = {}
        for name, group in data.groupby('Parents'):
            common_indices = group.index.intersection(missclass)
            print(f"ALL {name} :",len(group.index))
            # if group have the Whole_Class column, then check the Whole_Class
            if 'Whole_Class' in group.columns:
                Fast_baseline = len(group.loc[group['Whole_Class'] == 'FAST'].index)/len(group.index)
                Slow_baseline = len(group.loc[group['Whole_Class'] == 'SLOW'].index)/len(group.index)
            else:
                Fast_baseline = len(group.loc[group['Class'] == 'FAST'].index)/len(group.index)
                Slow_baseline = len(group.loc[group['Class'] == 'SLOW'].index)/len(group.index)    
            if Fast_baseline > Slow_baseline:
                print(f"baseline for {name} FAST:",Fast_baseline)
            else:   
                print(f"baseline for {name} SLOW:",Slow_baseline)
            print(f"length of {name}:",len(group.index))
            print(f"wrong predict for {name} :",len(group.loc[common_indices].index))
            dic[name] = {'all_group_count':len(group.index),
                         'wrong':len(group.loc[common_indices].index),
                         'accuracy':1 - len(group.loc[common_indices].index)/len(group.index),
                         'baseline':max(Fast_baseline,Slow_baseline)}
        return dic  
    def MCC(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def class_probabilities(classifier, X, y_true, parent_class=False):
        class_probabilities = classifier.predict_proba(X)
        if parent_class:
            columns = ['C10', 'C11', 'WB']
        else:
            columns = ['MEDIUM_pro', 'SLOW_pro']
        return pd.DataFrame(class_probabilities, columns=columns, index=y_true.index)
    # Dictionary to store additional custom metrics
    custom_metrics = {}

    @classmethod
    def add_custom_metric(cls, name, func):
        """Add a new custom metric to the metrics dictionary."""
        cls.custom_metrics[name] = func

    @classmethod
    def calculate_metrics(cls, y_true, y_pred, **kwargs):
        # cls means the class itself
        """Calculate all metrics including standard and custom ones."""
        test_data = kwargs.get('test_data',None)
        y_pred_proba = kwargs.get('y_pred_proba', None)  # Add this line
        # feature importance = kwargs.get('feature_importance', None)  # Add this line

        metrics = {
            'accuracy': cls.accuracy(y_true, y_pred),
            'baseline': cls.baseline(y_true,y_pred),
            'classification_report': cls.classification_report(y_true, y_pred),
            'AUC_ROC': cls.AUC_ROC(y_true, y_pred_proba) if y_pred_proba is not None else None,
            'MCC':cls.MCC(y_true, y_pred),
            'confusion_matrix': cls.confusion_matrix(y_true, y_pred),
            'misclassified_indices': cls.misclassified_indices(y_true, y_pred),
            'population_accuracy': cls.population_accuracy(y_true, y_pred, test_data),
            'feature_importance': kwargs.get('feature_importance', None),
        }
        

        # Calculate custom metrics
        for name, func in cls.custom_metrics.items():
            metrics[name] = func(y_true, y_pred, **kwargs)
        
        return metrics



def save_model_results(name, results_total):
    try:
        with open(name, 'wb') as f:
            pickle.dump(results_total, f)
        print(f"File saved successfully as '{name}' in the current directory.")
        print(f"Full path: {os.path.abspath(name)}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

    if os.path.exists(name):
        print(f"File exists: {name}")
    else:
        print(f"File does not exist: {name}")

def load_model_results(self):
    with open('model_results.pkl', 'rb') as f:
        results_total = pickle.load(f)
    return results_total


class Run_Main :
    def initialize_results(self, classifiers):
        """
        Initialize the results dictionary with empty lists for each classifier.
        
        Args:
            classifiers (dict): {'name': classifier_instance}
            
        Returns:
            dict: Dictionary with empty lists for each classifier
        """
        results = {
            'ImputationClassRecord': [],
            'params': {},  # params for the run
            'split_index': {'RandomState': 0, 'Train': [], 'Test': []}, 
            'selected_features': [], 
            'fs_size': [],  # size of feature subset
            'scores': []  # scores of the selected features
        }
        
        # Initialize results for each classifier
        for clf_name in classifiers.keys():
            results[clf_name] = []
            
        return results

    def __init__(self, snapper, location,X= None ,Y = None,DMFS = False, Metrics_classifiers = {'SVM': SVC(kernel='rbf', probability=True)},Feature_selection : str = None):
        self.Data = snapper
        if X.columns[0] == 'Parents':
            X['Parents'] = X['Parents'].map({'Class 10': 0, 'Class 11': 1, 'Wild Brood': 2}).values
        self.X = X
        self.Y = Y
        self.location = location
        self.DMFS = DMFS
        self.Metrics_classifiers = Metrics_classifiers
        # always do the imputation first
        self.Imputation = True
        self.Feature_selection = Feature_selection
        self.doGA = False
        self.split_index = {'RandomState':0 ,'Train':[],'Test':[],'Train_x':[],'Test_x':[],'Train_y':[],'Test_y':[],'train_Parents':[],'test_Parents':[]}
        # initialize the results
        self.randomState = 0
        self.results_total = self.initialize_results(Metrics_classifiers)
        self.add_params()

    def add_params(self):
        # Imputation true means do the DMFS
        self.params  = {
            'DMFS': self.DMFS,
            'Imputation': self.Imputation,
            'Feature_selection': self.Feature_selection,
            'doGA': self.doGA,
        }

        self.results_total['params'] = self.params
    
   

   
    def Run_Imputation(self,X_train, X_test, split = True,Train_indices = None,Test_indices = None,X = None):
        '''
        Run the imputation using the DK_KNN method.
        X_train: {dataframe} the training data
        X_test: {dataframe} the test data
        split: {bool} if True, then the input data is X_train, X_test. don't need to concat.
        Train_indices: {list} the indices of the training data, if split is True, then it is not needed.
        Test_indices: {list} the indices of the test data, if split is True, then it is not needed.
        
        '''
        # record the running time for the imputation
        start_time = time.time()
        X_train_without_impute = X_train.copy()
        X_test_without_impute = X_test.copy()

        RUN_DMimputation = Run_DMimputation(X, self.location,self.DMFS)
        X_whole_fine = RUN_DMimputation.run(split = split)
        RUN_DMimputation = Run_DMimputation(X_train, self.location,self.DMFS)
        X_train = RUN_DMimputation.run(split = split)
        # X_test without do the DMFS
        X_test = RUN_DMimputation.transform(X_test)
        end_time = time.time()
        print(f"Imputation completed in {(end_time - start_time):.2f} seconds") 

        # X_test = Run_DMimputation(X_test, location,False).run(split = split)
        print('X_train shape:',X_train.shape)
        print('X_test shape:',X_test.shape)
        # add the RUN_DMimputation to the results_total
        # self.results_total['ImputationClassRecord'].append(RUN_DMimputation)
        X_test = self.reorder_columns(X_test,X_train)
        print('after reorder X_train shape:',X_test.shape)
        # check uisng the sum of the isnull values
        if X_train.isnull().values.sum() > 0 or X_test.isnull().values.sum() > 0:
            print('X_train:',X_train.isnull().values.sum())
            print('X_test:',X_test.isnull().values.sum())
            # check where the NaN value is 
            print('X_train NaN value:',X_train[X_train.isnull().any(axis=1)])
            print('X_test NaN value:',X_test[X_test.isnull().any(axis=1)])
            # raise ValueError('Data have NaN value')
            sys.exit('Data have NaN value')
        # convert to unit8 for memory efficiency (after imputation validation)
        X_train = X_train.astype(np.uint8)
        X_test = X_test.astype(np.uint8)
        return X_train, X_test
    

    def feature_selection(self,X_train, X_test, y_train, y_test,classifier,subsetsize,GAparams: Dict[str, Any] = None):
        fs = FeatureSelection(estimator=classifier)
        fs.X_train = X_train
        fs.y_train = y_train
        fs.X_test = X_test
        fs.y_test = y_test
        # pass size of the total features
        fs.feature_size = X_train.shape[1]
        # get the distribution of the feature selection methods
        if self.Feature_selection != 'raw'or self.Feature_selection  != 'MULTI_GP':
            fs.analyze_feature_importance_with_elbows(X=X_train,y = y_train,method = self.Feature_selection,Fold = self.fold,Random_state = self.randomState)
        
        if self.Feature_selection== 'Chi2':
            start_time = time.time()
            X_train, X_test, selected_features, scores = fs.Chi2(X_train, y_train, X_test, y_test,subsetsize)
            end_time = time.time()
            print(f"Chi2 completed in {(end_time - start_time):.3f} seconds")
            # plot the elbow plot
            # fs.chi_squared_elbow_plot(X_train, y_train, X_test, y_test)
        elif self.Feature_selection == 'CMIM':
            # record the time for runnin the CMIM
            start_time = time.time()
            X_train, X_test, selected_features, scores = fs.CMIM(X_train, y_train, X_test, y_test,subsetsize)
            end_time = time.time()
            print(f"CMIM completed in {(end_time - start_time)/3600:.2f} hours")
        elif self.Feature_selection == 'MI':
            X_train, X_test, selected_features, scores = fs.MI(X_train, y_train, X_test, y_test,subsetsize)   
        elif self.Feature_selection == 'MULTI_GP':
            start_time = time.time()
            X_train, X_test, selected_features, scores = fs.MULTI_GP(X_train, y_train, X_test, y_test,subsetsize)   
            end_time = time.time()
            print(f"MULTI_GP completed in {(end_time - start_time)/3600:.2f} hours")
        elif self.Feature_selection == 'Relief':
            start_time = time.time()
            X_train, X_test, selected_features, scores = fs.Relief(X_train, y_train, X_test, y_test,subsetsize)     
            end_time = time.time()
            print(f"Relief completed in {(end_time - start_time)/3600:.2f} hours")   
        elif self.Feature_selection =='Relieff':
            X_train,X_test,selected_features,scores = fs.Relief_f(X_train, y_train, X_test, y_test,subsetsize)
        if self.doGA:
            selected_features = fs.genetic_algorithm(X_train, y_train,selected_features,GAparams['n_features'])
            # print('GA_param',fs.GA_params)
            X_train = fs.access_data(X_train,selected_features)
            X_test = fs.access_data(X_test,selected_features)
        return X_train,X_test,selected_features,scores    
    
    def reorder_columns(self, df_to_reorder, reference_df):
        """
        Reorder the columns of df_to_reorder to match the order of reference_df.
        If columns are missing from df_to_reorder, they will be ignored.

        Parameters:
        df_to_reorder (pd.DataFrame): The DataFrame whose columns need to be reordered.
        reference_df (pd.DataFrame): The DataFrame with the desired column order.

        Returns:
        pd.DataFrame: A new DataFrame with reordered columns.
        """
        # Get the column order from the reference DataFrame
        desired_order = reference_df.columns.tolist()

        # Find columns that exist in both DataFrames
        common_columns = [col for col in desired_order if col in df_to_reorder.columns]

        # Create a new DataFrame with columns reordered
        reordered_df = df_to_reorder[common_columns]

        # Check if any columns were missing and log a warning if so
        missing_columns = set(desired_order) - set(df_to_reorder.columns)
        if missing_columns:
            import warnings
            warnings.warn(f"Columns {missing_columns} are missing from df_to_reorder and will be ignored.")

        return reordered_df
    
    def remove_duplicate_columns(self, df):
        # Only remove columns where ALL values are the same
        non_constant_cols = df.columns[df.nunique() > 1]
        df_filtered = df[non_constant_cols]
        
        removed_count = df.shape[1] - df_filtered.shape[1]
        print(f"Removed {removed_count} constant columns")
        return df_filtered
    
    def evaluate_parent_prediction_svm(self, features_train, features_val, y_parents_train, y_parents_val):
        """Using Logistic regression to evaluate parent prediction capability
        """
        # using the logistic regression to evaluate the parent prediction capability
        from sklearn.linear_model import LogisticRegression
        # using desion tree 
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, classification_report
         
        print("\nEvaluating parent prediction capability using Logistic Regression...")

        train_features = features_train
        val_features = features_val

        lg = DecisionTreeClassifier()

        lg.fit(train_features, y_parents_train)
        train_pred = lg.predict(train_features)
        val_pred = lg.predict(val_features)

        train_acc = accuracy_score(y_parents_train, train_pred)
        val_acc = accuracy_score(y_parents_val, val_pred)

        train_report = classification_report(y_parents_train, train_pred)

        val_report = classification_report(y_parents_val, val_pred)
        # get the importance of the features
        print('feature importance:',lg.feature_importances_)  
        print("\nParent Classification Results:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("\nTraining Set Classification Report:")
        print(train_report)
        print("\nValidation Set Classification Report:")
        print(val_report)

        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_predictions': train_pred,
            'val_predictions': val_pred,
            'train_report': train_report,
            'val_report': val_report
        }
    

    def get_memory_usage(self):
        '''get current memory usage in MB'''
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return memory_mb
    
    def print_memory_usage(self,label=""):
        """Print current memory usage"""
        memory_mb = self.get_memory_usage()
        print(f"Memory usage {label}: {memory_mb:.2f} MB")
    def run_by_Classifier(self,X_train, X_test, y_train, y_test,classifierName,clf,FsName,SnapperData,CorrectPro = None,PraentClass = False,**kwargs):
        '''
        feature selection result is numpy, the raw is the dataframe
        X_train: {numpy}, shape (n_samples, n_features)
        input train data
        y_train: {numpy}, shape (n_samples,)
        input train class labels
        X_test: {numpy}, shape (n_samples, n_features)
        input test data
        y_test: {numpy}, shape (n_samples, )
        input test class labels
        classifierName: {str} name of the classifier
        clf: {classifier} classifier object
        FsName: {str} name of the feature selection method
        SnapperData: {dataframe} the data with the medium_pro
        '''

        y_parents_train = kwargs.get('y_parents_train',None)
        y_parents_test = kwargs.get('y_parents_test',None)

        # Check if model has random_state or random_seed parameter and set it
        if hasattr(clf, 'random_state'):
            clf.random_state = self.split_index['RandomState']
        elif hasattr(clf, 'random_seed'):
            clf.random_seed = self.split_index['RandomState']

        # check classifier allow to using the calibratedClassifierCV or not
        if classifierName in ['SVM', 'DWD']:
            calibrated_clf = CalibratedClassifierCV(clf, cv=5)
            # Set random_state for CalibratedClassifierCV if available
            if hasattr(calibrated_clf, 'random_state'):
                calibrated_clf.random_state = self.split_index['RandomState']
        else:
            calibrated_clf = clf

        parents_eval_LG= None

        print(f'=================classifierName{classifierName}===============')

        if classifierName.startswith('GP') :
            
            calibrated_clf.fit(X_train, y_train,fold = self.fold,y_parents_train = y_parents_train,y_parents_test = y_parents_test)
            # this y_parents_eval is using the feature matrix to predicted parents class, using SVM
            y_pred,Best_GP_,logbook = calibrated_clf.predict(X_test,y_true = y_test)
            y_pred_train,Best_GP_train,logbook = calibrated_clf.predict(X_train)
      
        else:    
            # check if the calibrated_clf have the fold_ attribute, then pass it to the fit method. (For the code written by myself)
            if hasattr(calibrated_clf, 'fold_'):
                calibrated_clf.fit(X_train, y_train,fold = self.fold)   

            elif classifierName.startswith('XGBoost'):
                # check if the y_train and y_test is string, then convert to 0,1. XGBoost only accept 0,1 as the label
                    # check the type of the y_train and y_test
                    y_train_index = y_train.index if hasattr(y_train, 'index') else None
                    y_test_index = y_test.index if hasattr(y_test, 'index') else None
                    # print('y_train type:',type(y_train))
                    y_train = np.where(y_train == 'FAST', 1, 0)
                    y_test = np.where(y_test == 'FAST', 1, 0)
            else: 
                 calibrated_clf.fit(X_train, y_train) 

            y_pred = calibrated_clf.predict(X_test)
            y_pred_train = calibrated_clf.predict(X_train)
            # added the acc and MCC to the list
        # check if y_parents_train and y_parents_test is not None, then evaluate the parent prediction capability
        if y_parents_train is not None and y_parents_test is not None:
            parents_eval_LG = self.evaluate_parent_prediction_svm(
                            X_train, X_test, y_parents_train, y_parents_test)
        else:
            parents_eval_LG = None
        print("feature seleciton : ",self.Feature_selection,type(self.Feature_selection))

        probabilities = calibrated_clf.predict_proba(X_test)
        probabilities_train = calibrated_clf.predict_proba(X_train)
        if classifierName.startswith('XGBoost'):
            y_test = np.where(y_test == 1, 'FAST', 'SLOW')
            y_pred = np.where(y_pred == 1, 'FAST', 'SLOW')
            y_train = np.where(y_train == 1, 'FAST', 'SLOW')
            y_pred_train = np.where(y_pred_train == 1, 'FAST', 'SLOW')
            if y_test_index is not None:
                y_pred = pd.Series(y_pred, index=y_test_index)
                y_test = pd.Series(y_test, index=y_test_index)
            if y_train_index is not None:
                y_pred_train = pd.Series(y_pred_train, index=y_train_index)
                y_train = pd.Series(y_train, index=y_train_index)
        # calculate the metrics 
        metrics_kwargs = {'test_data': SnapperData.loc[y_test.index], 'y_pred_proba': probabilities,'feature_importance':calibrated_clf.feature_importances_ if hasattr(calibrated_clf, 'feature_importances_') else None}
        metrics_kwargs_train = {'test_data': SnapperData.loc[y_train.index], 'y_pred_proba': probabilities_train,'feature_importance':calibrated_clf.feature_importances_ if hasattr(calibrated_clf, 'feature_importances_') else None}
        print("len y_pred:",len(y_pred),len(y_test))
        # if y_test and y_pred is 0,1 then convert to the SLOW and FAST

        test_metrics = Metrics.calculate_metrics(y_test, y_pred, **metrics_kwargs)
        train_metrics = Metrics.calculate_metrics(y_train, y_pred_train, **metrics_kwargs_train)
      
        
        if self.Feature_selection == 'CMIM':
            print('test_metrics_CMIM:',test_metrics)

        # check the shape below 'Model_Pro': probabilities[:,0], 'Predicted': y_pred, 'True': y_test, 'True_prob': SnapperData.loc[y_test.index,'medium_pro'],
        print('probabilities shape:',probabilities.shape)
        print('y_pred shape:',len(y_pred))
        print('test_metrics:',test_metrics)
                 
        # None if parents_eval_LG != None else  parents_eval_LG['train_predictions'] if len(y_pred)>300 else parents_eval_LG['val_predictions']
        def Eval_check (parents_eval_LG = parents_eval_LG):
            if parents_eval_LG is not None:
                return parents_eval_LG['train_predictions'] if len(y_pred)>300 else parents_eval_LG['val_predictions']
            else:
                return None

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model_Pro': probabilities[:,0]if probabilities.ndim > 1 else probabilities,
            # 'Model_Pro': probabilities,
            'Predicted': y_pred,
            'Eval_Parents': Eval_check(),
            'Eval_Accuracy': parents_eval_LG['val_accuracy'] if parents_eval_LG is not None else None,
            'True': y_test,
            'True_prob': SnapperData.loc[y_test.index,'medium_pro'] if 'medium_pro' in SnapperData.columns else None,

        }, index=y_test.index)

        print("Evall_Parents:",results_df['Eval_Parents'])
        print("check probabilityes : ",probabilities[:,0]if probabilities.ndim > 1 else probabilities)
        
        


        # Prepare the results dictionary
        results = {
            'classifier_name': classifierName,
            'Feature_selection': FsName,
            'Imputation': self.Imputation,
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'results_df': results_df,
            'misclassified_test': results_df.loc[test_metrics['misclassified_indices']], 
            'parents_eval_LG': parents_eval_LG,
            'f1_eval_LG':f1 if 'f1' in locals() else None,
            'acc_eval_LG':acc if 'acc' in locals() else None,
            'Param_NN':  getattr(calibrated_clf, '_get_param', lambda: None)(), 
        }

        return results  
    

    
    def run_classification(self,fs_size = 0,results_total = {},randomState = 0):
        ''''This will run the classification for the given data(not Gene data) and classifiers.'''
        print('Feature selection method:',self.Feature_selection,"----DOGA:",self.doGA,'fs_size:',fs_size)
        X = self.X
        Y = self.Y
        self.fold = 1
        # set the golbal random state,not only np
        np.random.seed(randomState)
        random.seed(randomState)
        
        # using the normal Kfold
        kfold = KFold(n_splits=5, shuffle=True, random_state=randomState)
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
            print('Memory usage before imputation:',self.get_memory_usage())
            # ccheck the y_train and y_test's value counts to check the balance of the data
            print('Y_train value counts:',y_train.value_counts())
            print('Y_test value counts:',y_test.value_counts())
            # store the split index
            self.split_index['RandomState'] = randomState
            self.split_index['Train'].append(train_index)
            self.split_index['Test'].append(test_index)
            # chck if there are any NaN values in the data, if have, then run the imputation
            if X_train.isnull().values.any() or X_test.isnull().values.any():
                print('X_train:',X_train.isnull().values.any())
                print('X_test:',X_test.isnull().values.any())
                # run the imputation
                ############# raise ValueError('Data have NaN value') #############
                X_train, X_test = self.Run_Imputation(X_train, X_test,split = True)
                X_duplicate = pd.concat([X_train,X_test],axis=0)
                print('Duplicated columns in X_duplicate:',len(X_duplicate.columns[X_duplicate.nunique() <= 1]))
            print('Memory usage after imputation:',self.get_memory_usage())
            if self.Feature_selection is not None:
                if self.doGA:
                    # this classifier for the GA 
                    GAFSclassifier = self.Metrics_classifiers['SVM']
                else:
                    GAFSclassifier = None

                X_train, X_test,selected_features,scores = self.feature_selection(X_train, X_test, y_train, y_test,GAFSclassifier,fs_size,GAparams = {'n_features':500})   
                    
                print('selected_features:',selected_features)
                # results_total added selected features.
                self.results_total['selected_features'].append(selected_features)
                self.results_total['fs_size'].append(fs_size)
                self.results_total['scores'].append(scores)
    
            for classifier_name, classifier in self.Metrics_classifiers.items():
                if self.Feature_selection is not None : 
                    results = self.run_by_Classifier(X_train, X_test, y_train, y_test, classifier_name, classifier, self.Feature_selection, self.Data)
            
                if classifier_name not in self.results_total.keys():
                    self.results_total[classifier_name] = []
                self.results_total[classifier_name].append(results)
                print(f'Memory usage {self.fold} after running classifier:',self.get_memory_usage())
            self.fold += 1
        self.results_total['split_index'] = self.split_index
        print('Memory usage after all runs:',self.get_memory_usage())
        print('Results:',self.results_total)
        # save the results to the file
        return self.results_total    

    def run(self,fs_size = 0,randomState = 0):
        '''
        fs_size: {int} size of feature subset
        results_total: {dict} store the results of all classifiers
        randomState: {int} random state
        '''
        print('Feature selection method:',self.Feature_selection,"----DOGA:",self.doGA,'fs_size:',fs_size)
        X = self.X
        Y = self.Y
        self.fold = 1
        np.random.seed(randomState)
        print("random state : ",randomState)
        self.randomState = randomState
        # Using the all instance to caculate the chi2 top 4000
        sss = StratifiedKFold(n_splits= 5,shuffle = True, random_state = randomState)
        # sss = StratifiedShuffleSplit(n_splits= 2, test_si≠ze=0.2, random_state=randomState)

        for train_index, test_index in sss.split(X, self.Data.Parents):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            if 'Parents' in self.Data.columns:
                # get the paretns class
                y_parents_train ,y_parents_test= self.Data.iloc[train_index].Parents,self.Data.iloc[test_index].Parents
                y_growth_rate, y_growth_rate_test = self.Data.iloc[train_index].Growth_rate,self.Data.iloc[test_index].Growth_rate
                # caculate the Y train and Y test's value counts to check the balance of the data
                print('Y_train value counts:',y_train.value_counts())
                print('Y_test value counts:',y_test.value_counts())
                # print each parents class's value counts, train
                self.split_index['train_Parents'].append(y_parents_train)
                self.split_index['test_Parents'].append(y_parents_test)
                print("split_ parents_train",y_parents_train.value_counts())
            else :
                y_parents_train ,y_parents_test = None, None
            self.split_index['RandomState'] = randomState
            self.split_index['Train'].append(train_index)
            self.split_index['Test'].append(test_index)
            # print how many memory is used
            self.print_memory_usage(label="before imputation")
            
        
            if self.Imputation:
                X_train, X_test = self.Run_Imputation(X_train, X_test,split = True,Train_indices= train_index,Test_indices = test_index,X = self.X)
                # record imputation x
                self.split_index['Train_x'].append(X_train.columns)
                self.split_index['Train_y'].append(y_train)
                # ordered the columns
                cols = sorted(X_train.columns, key=lambda x: int(x.split('-')[-1]))
                X_train = X_train[cols]
                X_test = X_test[cols]
         
            # if feature selection methods is none, then ignore the FS
            if self.Feature_selection is not None:
                if self.doGA:
                    # this classifier for the GA 
                    GAFSclassifier = self.Metrics_classifiers['SVM']
                else:
                    GAFSclassifier = None
                logger.info(f'Feature selection method: {self.Feature_selection}, doGA: {self.doGA}, fs_size: {fs_size}')  
                X_train, X_test,selected_features,scores = self.feature_selection(X_train, X_test, y_train, y_test,GAFSclassifier,fs_size,GAparams = {'n_features':500})   
                
                # results_total added selected features.
                self.results_total['selected_features'].append(selected_features)
                self.results_total['fs_size'].append(fs_size)
                self.results_total['scores'].append(scores)

            for classifier_name, classifier in self.Metrics_classifiers.items():
                print(f'Running classifier: {classifier_name} with feature selection: {self.Feature_selection}')
                # start time 
                start_time = time.time()
                if self.Feature_selection is not None : 
                    logger.info(f'Running classifier: {classifier_name} with feature selection: {self.Feature_selection}')
                    results = self.run_by_Classifier(X_train, X_test, y_train, y_test, classifier_name, classifier, self.Feature_selection, self.Data,y_parents_train = y_parents_train,y_parents_test = y_parents_test)
                else:
                    if classifier_name.startswith('GenNN')or classifier_name.startswith('PCA'):
                        # print('parents:',y_parents_train)
                        results = self.run_by_Classifier(X_train, X_test, y_train, y_test, classifier_name, classifier, 'raw', self.Data, y_parents_train = y_parents_train,y_parents_test = y_parents_test)
                    else :  results = self.run_by_Classifier(X_train, X_test, y_train, y_test, classifier_name, classifier, 'raw', self.Data, y_parents_train = y_parents_train,y_parents_test = y_parents_test)
                if classifier_name not in self.results_total.keys():
                    self.results_total[classifier_name] = []
                #end time
                end_time = time.time()
                print(f'Classifier {classifier_name} finished in {end_time - start_time:.2f} seconds') 
                self.results_total[classifier_name].append(results)
                
            self.print_memory_usage(label=f"fold{self.fold} after classification")    
            self.fold += 1    
        
        self.results_total['split_index'] = self.split_index 
        print('Results:',self.results_total)       
        return self.results_total        
            
            
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

   