#%%
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from dwd.socp_dwd import DWD
from Classifiers_new import Run_Main
import sys
import time
# This runner file is using to reproduce the results of the paper
# nohup python -u runner.py 0 False False > runner0.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u runner.py >runner.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python runner.py
# results['Chi2']['split_index'].keys() get the RandomState = , 'Train_x' = [], 'Test_x' = [], 'Train_y', 'Test_y'
# write a quick run, have the information of the Train,test split. 
# for server in cuda0 cuda1 cuda2 cuda3 cuda4 cuda5 cuda6 cuda7 cuda8 cuda9 cuda10 cuda11 cuda12 cuda13 cuda14 cuda15 cuda16 gryphon red-tomatoes piccolo the-villa bordeaux
# do
# printf "------------%s------------\n" $server
# ssh $server 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
# done
class ExperimentRunner:
    def __init__(self, data_path,location_path,seed = 1):
        # '/home/chenze/Desktop/Code/RawRUN/Redata_withMissing.csv'
        # '/home/chenze/Desktop/Code/Snapper_SNP_locations.xlsx'
        # pd.read_csv('/home/chenze/Desktop/Code/RawRUN/snpchip_snps_SNAv2.csv')
        # '/home/chenze/Desktop/Code/Snapper_SNP_locations.xlsx'
        self.data_path = data_path
        self.location_path = location_path
        self.data = None
        self.seed = seed
        self.location = pd.read_excel(location_path)
        # self.location = pd.read_csv(location_path)
        self.metrics_classifiers = self._setup_classifiers()
        self.metrics_regressors = self._setup_regressors()  # 添加回归器
        self.fs_params = self._setup_fs_params()
        
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def _setup_classifiers(self):
        return {
            'SVM': SVC(kernel='rbf',probability=True,random_state=self.seed),
            'DWD': DWD(),
            'LR': LogisticRegression(random_state=self.seed),
            'RF': RandomForestClassifier(random_state=self.seed),
            'NB': GaussianNB(), 
            # 'LR': LogisticRegression(),
            # 'SVM': SVC(kernel='rbf', probability=True),
            # 'DWD': DWD(),
            # 'RF': RandomForestClassifier(),
            # 'NB': GaussianNB(),
            # "GP": GPBaseModel(),
           
        }

    def _setup_regressors(self):
        """Setup regression models"""
        return {
            'SVR': SVR(kernel='rbf'),
            'RF': RandomForestRegressor(),
            'Ridge': Ridge(),
            'Lasso': Lasso()
        }

    def _setup_fs_params(self):
        # print [4500,1100,1400,1400]
        # running fast first. To check the results and ouput is correct.

        return {
            'raw': 0,
            'Chi2':3500,#4000
            'CMIM':600, # 1000 
            'MI':700,#1200
            'Relief':1200,#1000
        }

    def load_data(self):
        self.data = pd.read_csv(self.data_path, index_col=0)
        # self.location = pd.read_excel(self.location_path)
        # self.location = pd.read_csv(self.location_path)
        print("Data loaded successfully \n", self.data, "\n", self.location.head())

    def run_experiments(self, random_state, dmfs, do_ga, methods, export_data=False,name_s = None):
        if self.data is None:
            self.load_data()

        results_total = {}
        starttime = time.time()
        
        for method in methods:
            print(f"\nRunning {method} method...")
            methods_times = time.time()
            fs_method = None if method == 'raw' else method
            
            runner = Run_Main(self.data, self.location, 
                            DMFS=dmfs,
                            Y=self.data.Class,
                            X=self.data.iloc[:, 4:],
                            Metrics_classifiers=self.metrics_classifiers,
                            Feature_selection=fs_method)
            runner.doGA = do_ga
            
            results_total[method] = runner.run(
                fs_size=self.fs_params[method],
                randomState=random_state
            )
            # how many hours = (time.time()-starttime)/3600
            print(f"Method {method} finished in {(time.time()-methods_times)/3600} seconds")
          
        if export_data:
            filename = f'Results_{"DMFS" if dmfs else "noDMFS"}_{"GA" if do_ga else "noGA"}_{random_state}{name_s}.pkl'
            self.save_model_results(filename, results_total)
        
        #total time
        print(f'Total time: {(time.time()-starttime)/3600} hours')
        return results_total

    @staticmethod
    def save_model_results(name, results):
        import pickle
        import os
        
        try:
            with open(name, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved successfully as '{name}'")
            print(f"Full path: {os.path.abspath(name)}")
        except Exception as e: 
            print(f"Error saving results: {str(e)}")

    def run_interactive(self, random_state=42, dmfs=False, do_ga=False, methods=['Chi2'], export_data=False,name_s =''):
        print(f"Starting experiment with parameters:")
        print(f"Random State: {random_state}")
        print(f"DMFS: {dmfs}")
        print(f"GA: {do_ga}")
        print(f"Methods: {methods}")
        return self.run_experiments(random_state, dmfs, do_ga, methods, export_data,name_s)

    def quick_run(self, method='Chi2', export_data=False,random_state=42,name_s = None):
        return self.run_interactive(methods=[method], export_data=export_data,random_state=random_state,name_s = name_s)

    def run_all(self):
        methods = ['raw', 'Chi2', 'CMIM', 'MI', 'Relief', 'Relieff']
        return self.run_interactive(methods=methods)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("RandomState", type=int, help="Random state for reproducibility")
    parser.add_argument("DMFS", type=ExperimentRunner.str2bool, help="Whether to use DMFS")
    parser.add_argument("doGA", type=ExperimentRunner.str2bool, help="Whether to use GA")
    parser.add_argument("--methods", nargs='+', 
                       default=['raw','Chi2','MI'],
                       choices=['raw', 'Chi2', 'CMIM', 'MI', 'Relief'],
                       help="Feature selection methods to use")
    args = parser.parse_args()

    print(f"Running experiment with parameters:random_state={args.RandomState}, DMFS={args.DMFS}, doGA={args.doGA}, methods={args.methods}")
    runner = ExperimentRunner(seed=args.RandomState,data_path='/home/chenze/Desktop/Code/RawRUN/Redata_withMissing.csv',location_path='/home/chenze/Desktop/Code/Snapper_SNP_locations.xlsx')
    results = runner.run_experiments(args.RandomState, args.DMFS, args.doGA, args.methods, export_data=True)
    return results

def get_acc(results,classifiers,fs = 'raw'):
    for classifier in classifiers:
        print("Classifier:",classifier)
        Acc = []
        MCC = []
        for i in range(5):  
            accuracy = results[fs][classifier][i]['test_metrics'] 
            print("ACC :" ,accuracy['accuracy'],"  MCC:", accuracy['MCC'])
            Acc.append(accuracy['accuracy'])
            MCC.append(accuracy['MCC'])
            np.mean(Acc)
            np.mean(MCC)
        print("ACC mean:",np.mean(Acc),"  MCC mean:",np.mean(MCC))
def quick_run_no_feature_selection(export_data=False,name_s = 'a'):
    runner = ExperimentRunner()
    return runner.quick_run(method='raw', export_data=export_data,name_s = name_s),runner.metrics_classifiers.keys()


def quick_run(method='Chi2', export_data=False):
    runner = ExperimentRunner()
    return runner.quick_run(method=method, export_data=export_data),runner.metrics_classifiers.keys()

def quick_run_regression(export_data=False, name_s='regression_test'):
    """Quick function to run regression analysis"""
    runner = ExperimentRunner()
    return runner.run_regression(
        methods=['raw'], 
        export_data=export_data, 
        name_s=name_s
    ), runner.metrics_regressors.keys()

def run_all(export_data=False):
    runner = ExperimentRunner()
    return runner.run_all()
def get_data():
    runner = ExperimentRunner()
    runner.load_data()
    return runner.data, runner.location


#%%

if __name__ == '__main__':
    main()


# Usage example:
# python runner.py 42 True True --methods raw Chi2 CMIM MI Relief Relieff
# results, classifier = quick_run_no_feature_selection(True,'GenNN')




