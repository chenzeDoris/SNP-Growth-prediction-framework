import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from typing import Union, Callable, List
import matplotlib.pyplot as plt
from itertools import combinations
from deap import base, creator, tools, algorithms
import random
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from skfeature.function.information_theoretical_based import CMIM, MRMR, FCBF, CIFE,MIFS
from sklearn.metrics import make_scorer, matthews_corrcoef
import sklearn_relief as relief
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import time
import os
from kneed import KneeLocator
import sys


class EvaluationFunction(ABC):
    def __init__(self, estimator: BaseEstimator, cv: int = 5):
        self.estimator = estimator
        self.cv = cv

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> float:
        pass

class AccuracyEvaluation(EvaluationFunction):
    def evaluate(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> float:

        return np.mean(cross_val_score(self.estimator, X, y, cv=self.cv, scoring='accuracy'))
class MCC_Evaluation(EvaluationFunction):
    def evaluate(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> float:
        # Create a scorer using MCC
        mcc_scorer = make_scorer(matthews_corrcoef)
        # Use cross_val_score with MCC
        scores = cross_val_score(model, X, y, cv=5, scoring=mcc_scorer)

        return np.mean(scores)


class FeatureSelection:
    def __init__(self, estimator: BaseEstimator, cv: int = 5):
        self.estimator = estimator
        self.eval_function = AccuracyEvaluation(estimator)
        self.cv = cv
        self.best_features = None
        self.best_score = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_size = None
        self.GA_params = {'population_size': None, 'generations': None, 'crossover_prob': None, 'mutation_prob': None}
        # show the all the feature_importance first



    def detect_elbow_points(self, scores, method='kneedle', n_points=5):
        """
        Detect elbow points in feature importance scores using multiple methods.
        
        Parameters:
        -----------
        scores : array-like
            Feature importance scores (sorted in descending order)
        method : str
            Method for elbow detection: 'kneedle', 'second_derivative', 'difference'
        n_points : int
            Number of elbow points to return
        
        Returns:
        --------
        elbow_indices : list
            Indices of detected elbow points
        """
        
        if len(scores) < 20:
            return [len(scores) // 4, len(scores) // 2, len(scores) * 3 // 4]
        
        if method == 'kneedle':
            # Use kneed library for Kneedle algorithm
            x = np.arange(len(scores))
            y = np.array(scores)
            
            try:
                kl = KneeLocator(x, y, curve='convex', direction='decreasing', S=1.0)
                
                if kl.knee is not None:
                    elbow_indices = [kl.knee]
                    
                    # Find additional knees by trying different sensitivity values
                    for sensitivity in [0.5, 2.0]:
                        kl_alt = KneeLocator(x, y, curve='convex', direction='decreasing', S=sensitivity)
                        if kl_alt.knee is not None and kl_alt.knee not in elbow_indices:
                            elbow_indices.append(kl_alt.knee)
                else:
                    elbow_indices = []
                    
            except:
                elbow_indices = []
            
            # If not enough points found, fallback to finding peaks in differences

            if len(elbow_indices) < n_points:
                diff = np.diff(scores)
                abs_diff = np.abs(diff)
                peaks, _ = find_peaks(abs_diff, distance=len(scores)//20)
                sorted_peaks = sorted(peaks, key=lambda x: abs_diff[x], reverse=True)
                elbow_indices.extend(sorted_peaks[:n_points-len(elbow_indices)])
        
        elif method == 'second_derivative':
            # Apply smoothing to reduce noise
            sigma = max(1, len(scores) / 200)
            smoothed = gaussian_filter1d(scores, sigma=sigma)
            
            # Calculate second derivative
            first_derivative = np.gradient(smoothed)
            second_derivative = np.gradient(first_derivative)
            
            # Find peaks in absolute second derivative
            abs_second_deriv = np.abs(second_derivative)
            height_threshold = np.mean(abs_second_deriv) + np.std(abs_second_deriv)
            min_distance = max(20, len(abs_second_deriv) // 50)
            
            peaks, _ = find_peaks(abs_second_deriv, height=height_threshold, distance=min_distance)
            
            if len(peaks) > 0:
                # Sort by second derivative magnitude
                peak_values = [(peak, abs_second_deriv[peak]) for peak in peaks]
                peak_values.sort(key=lambda x: x[1], reverse=True)
                elbow_indices = [peak for peak, _ in peak_values[:n_points]]
            else:
                # Fallback: highest second derivative points
                sorted_indices = np.argsort(abs_second_deriv)[::-1]
                elbow_indices = sorted_indices[:n_points].tolist()
        
        elif method == 'difference':
            # Calculate first differences
            first_diff = np.abs(np.diff(scores))
            
            # Find points with maximum rate of change
            sorted_diff_indices = np.argsort(first_diff)[::-1]
            max_diff_points = sorted_diff_indices[:n_points].tolist()
            
            # Find points where differences drop significantly
            threshold = 0.1 * np.median(first_diff)
            below_threshold = np.where(first_diff < threshold)[0]
            
            # Combine both approaches
            elbow_indices = max_diff_points
            if len(below_threshold) > 0:
                elbow_indices.extend(below_threshold[:n_points//2])
        
        else:
            raise ValueError("Method must be 'kneedle', 'second_derivative', or 'difference'")
        
        # Ensure we have at least some elbow points and they're reasonable
        if not elbow_indices or len(elbow_indices) == 0:
            # Fallback: use logarithmic spacing
            log_points = np.logspace(np.log10(100), np.log10(len(scores)-1), n_points, dtype=int)
            elbow_indices = log_points.tolist()
        
        # Remove duplicates and sort
        elbow_indices = sorted(list(set(elbow_indices)))
        
        # Ensure indices are within bounds
        elbow_indices = [idx for idx in elbow_indices if 0 <= idx < len(scores)]
        
        return elbow_indices[:n_points]
    
    def analyze_feature_importance_with_elbows(self, X, y, method='Chi2', 
                                            outdir='./FS_Dis/', plot_distributions=True, 
                                            detect_elbows=True,Fold = None,Random_state = None):
        """
        Analyze feature importance with elbow point detection for a single method.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Input features (SNP data)
        y : array-like
            Target variable (phenotype)
        method : str
            Method to analyze: 'chi2', 'mi', 'relief', 'cmim'
        outdir : str
            Output directory for saving plots
        plot_distributions : bool
            Whether to plot distribution graphs
        detect_elbows : bool
            Whether to detect and report elbow points
        
        Returns:
        --------
        results : dict
            Dictionary containing scores, elbow points, and recommendations for the method
        """
        
        # Create output directory if it doesn't exist
        os.makedirs(outdir, exist_ok=True)
        
        max_features = self.feature_size
        
        print(f"\n{'='*50}")
        print(f"Analyzing {method.upper()} method...")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Calculate feature importance scores
        if method == 'Chi2':
            selector = SelectKBest(chi2, k=max_features)
            selector.fit(X, y)
            scores = selector.scores_
            method_name = "Chi-Square"
            
        elif method == 'MI':
            # check y value is continous or categorical
            selector = SelectKBest(mutual_info_classif, k=max_features)
            selector.fit(X, y)
            scores = selector.scores_
            method_name = "Mutual Information"
            
        elif method == 'Relief':
            # Assuming you have relief imported
            fs = relief.Relief(n_features=X.shape[1], random_state=3)
            fs.fit(X.values, y.values)
            feat_indices = fs.w_
            scores = max(feat_indices) - feat_indices + 1
            method_name = "Relief"
            
        elif method == 'CMIM':
            return None
        else:
            raise ValueError("Method must be one of: 'chi2', 'mi', 'relief', 'cmim'")
        
        print(f"Feature importance calculation completed in {time.time() - start_time:.2f} seconds")
        
        # Sort scores
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        
        # Detect elbow points using multiple methods
        elbow_points = []
        if detect_elbows:
            print("Detecting elbow points using multiple algorithms...")
            
            # Apply different elbow detection methods
            # 'kneedle', 'second_derivative', or 'difference'
            knee_elbows = self.detect_elbow_points(sorted_scores, method='kneedle', n_points=4)
            print(f"  Knee method detected: {[ep + 1 for ep in knee_elbows]}")

            # Combine all detected elbow points
            all_elbows = set(knee_elbows)
            elbow_points = sorted(list(all_elbows))
            
            # Keep the most significant ones (limit to top 8)
            if len(elbow_points) > 8:
                # Score each elbow point by how many methods detected it
                elbow_scores = {}
                for ep in elbow_points:
                    score = 0
                    if ep in knee_elbows:
                        score += 3  # Knee method gets higher weight
                    elbow_scores[ep] = score
                
                # Keep top scored elbow points
                sorted_elbows = sorted(elbow_scores.items(), key=lambda x: x[1], reverse=True)
                elbow_points = [ep for ep, _ in sorted_elbows[:8]]
                elbow_points.sort()
            
            print(f"\nFinal combined elbow points at indices: {elbow_points}")
            print(f"Corresponding feature counts: {[ep + 1 for ep in elbow_points]}")
        
        # Plotting
        if plot_distributions:
            # Create comprehensive plot
            plt.figure(figsize=(16, 12))
            
            # Subplot 1: Full distribution with elbow points
            plt.subplot(2, 3, 1)
            plt.plot(np.arange(len(sorted_scores)), sorted_scores, 'b-', alpha=0.7, linewidth=1.5)
            if elbow_points:
                plt.scatter(elbow_points, sorted_scores[elbow_points], color='red', s=80, 
                        zorder=5, label=f'Elbow Points ({len(elbow_points)})', alpha=0.8)
                # Add annotations for first few elbow points
                for i, ep in enumerate(elbow_points[:5]):
                    plt.annotate(f'{ep+1}', (ep, sorted_scores[ep]), xytext=(5, 5), 
                            textcoords='offset points', fontsize=9, color='red', fontweight='bold')
            plt.title(f"Full Distribution - {method_name}")
            plt.xlabel("Feature Rank")
            plt.ylabel("Importance Score")
            if elbow_points:
                plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Top 10000 features (zoomed view)
            plt.subplot(2, 3, 2)
            top_n = min(10000, len(sorted_scores))
            plt.plot(np.arange(top_n), sorted_scores[:top_n], 'b-', alpha=0.7, linewidth=1.5)
            visible_elbows = [ep for ep in elbow_points if ep < top_n]
            if visible_elbows:
                plt.scatter(visible_elbows, sorted_scores[visible_elbows], color='red', s=80, 
                        zorder=5, label=f'Elbow Points', alpha=0.8)
                for ep in visible_elbows[:5]:
                    plt.annotate(f'{ep+1}', (ep, sorted_scores[ep]), xytext=(5, 5), 
                            textcoords='offset points', fontsize=9, color='red', fontweight='bold')
            plt.title(f"Top {top_n} Features - {method_name}")
            plt.xlabel("Feature Rank")
            plt.ylabel("Importance Score")
            if visible_elbows:
                plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Cumulative importance
            plt.subplot(2, 3, 3)
            normalized_scores = sorted_scores / np.sum(sorted_scores)
            cumulative = np.cumsum(normalized_scores)
            plt.plot(np.arange(len(cumulative)), cumulative, 'g-', linewidth=2)
            
            # Add threshold lines and elbow points
            for threshold, color in [(0.8, 'r'), (0.9, 'orange'), (0.95, 'purple')]:
                plt.axhline(y=threshold, color=color, linestyle='--', alpha=0.5, 
                        label=f'{int(threshold*100)}% threshold')
                if np.any(cumulative >= threshold):
                    threshold_idx = np.argmax(cumulative >= threshold)
                    plt.axvline(x=threshold_idx, color=color, linestyle=':', alpha=0.3)
                    plt.text(threshold_idx + len(cumulative)*0.01, threshold-0.02, f'{threshold_idx+1}', 
                            fontsize=8, color=color, fontweight='bold')
            
            # Mark elbow points on cumulative curve
            if elbow_points:
                elbow_cumulative = cumulative[elbow_points]
                plt.scatter(elbow_points, elbow_cumulative, color='red', s=60, 
                        zorder=5, alpha=0.7)
            
            plt.title(f"Cumulative Importance - {method_name}")
            plt.xlabel("Number of Features")
            plt.ylabel("Cumulative Importance")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Rate of change (smoothed differences)
            plt.subplot(2, 3, 4)
            if len(sorted_scores) > 1:
                diffs = np.abs(np.diff(sorted_scores))
                # Apply smoothing
                sigma = max(1, len(diffs) / 200)
                smoothed_diffs = gaussian_filter1d(diffs, sigma=sigma)
                plt.plot(np.arange(len(smoothed_diffs)), smoothed_diffs, 'purple', linewidth=2)
                plt.yscale('log')
                plt.title(f"Rate of Change (Log Scale) - {method_name}")
                plt.xlabel("Feature Rank")
                plt.ylabel("Absolute Difference")
                plt.grid(True, alpha=0.3)
            
            # Subplot 5: Elbow detection visualization (distance from line)
            plt.subplot(2, 3, 5)
            if len(sorted_scores) > 1:
                normalized_scores_viz = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())
                x = np.arange(len(normalized_scores_viz))
                
                # Calculate distances for visualization
                line_start = [0, normalized_scores_viz[0]]
                line_end = [len(normalized_scores_viz) - 1, normalized_scores_viz[-1]]
                distances = []
                for i in range(len(normalized_scores_viz)):
                    x1, y1 = line_start
                    x2, y2 = line_end
                    x0, y0 = i, normalized_scores_viz[i]
                    distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                    distances.append(distance)
                
                plt.plot(x, distances, 'orange', linewidth=2, label='Distance from line')
                if elbow_points:
                    plt.scatter(elbow_points, np.array(distances)[elbow_points], color='red', s=60, 
                            zorder=5, label='Detected elbows')
                plt.title(f"Elbow Detection Visualization - {method_name}")
                plt.xlabel("Feature Rank")
                plt.ylabel("Distance from Baseline")
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Subplot 6: Log-scale view of top features
            plt.subplot(2, 3, 6)
            log_top_n = min(5000, len(sorted_scores))
            plt.plot(np.arange(log_top_n), sorted_scores[:log_top_n], 'b-', alpha=0.7, linewidth=1.5)
            plt.yscale('log')
            visible_elbows_log = [ep for ep in elbow_points if ep < log_top_n]
            if visible_elbows_log:
                plt.scatter(visible_elbows_log, sorted_scores[visible_elbows_log], color='red', s=80, 
                        zorder=5, label='Elbow Points', alpha=0.8)
            plt.title(f"Top {log_top_n} Features (Log Scale) - {method_name}")
            plt.xlabel("Feature Rank")
            plt.ylabel("Importance Score (Log Scale)")
            if visible_elbows_log:
                plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            # Save to specified directory
            plt.savefig(os.path.join(outdir, f"Feature_analysis_{method}_RS{Random_state}_Fold{Fold}.png"), 
                    dpi=300, bbox_inches='tight')
            # plt.show()
        
        # Detailed analysis and recommendations
        print(f"\n{method_name} Analysis Results:")
        print(f"Total features analyzed: {len(sorted_scores)}")
        
        # Calculate statistics at elbow points
        if elbow_points:
            print(f"\nDetailed analysis of {len(elbow_points)} elbow points:")
            normalized_scores = sorted_scores / np.sum(sorted_scores)
            cumulative = np.cumsum(normalized_scores)
            
            for i, ep in enumerate(elbow_points):
                feature_count = ep + 1
                score_at_elbow = sorted_scores[ep]
                cumulative_at_elbow = cumulative[ep]
                
                # Calculate score ratio compared to max
                score_ratio = score_at_elbow / sorted_scores[0] if sorted_scores[0] > 0 else 0
                
                print(f"\n  Elbow Point {i+1}: {feature_count} features")
                print(f"    Score: {score_at_elbow:.6f}")
                print(f"    Score ratio to max: {score_ratio:.4f}")
                print(f"    Cumulative importance: {cumulative_at_elbow*100:.2f}%")
                
                # Calculate relative importance drop
                if ep < len(sorted_scores) - 1:
                    next_score = sorted_scores[ep + 1]
                    relative_drop = (score_at_elbow - next_score) / score_at_elbow if score_at_elbow > 0 else 0
                    print(f"    Relative drop to next: {relative_drop*100:.2f}%")
        
        # Store results
        results = {
            'scores': sorted_scores,
            'indices': sorted_indices,
            'elbow_points': [ep + 1 for ep in elbow_points],  # Convert to 1-based indexing
            'method_name': method_name,
            'elbow_indices': elbow_points  # Keep 0-based for internal use
        }
        
        # Generate final recommendations based on detected elbow points
        print(f"\n{'='*60}")
        print("ELBOW POINT SUMMARY & RECOMMENDATIONS")
        print(f"{'='*60}")
        
        elbow_points_1based = results['elbow_points']
        
        print(f"\n{method.upper()}:")
        print(f"  Detected elbow points: {elbow_points_1based}")
        
        if elbow_points_1based:
            # Group elbow points into ranges for practical use
            small_range = [ep for ep in elbow_points_1based if ep <= 1500]
            medium_range = [ep for ep in elbow_points_1based if 1500 < ep <= 5000]
            large_range = [ep for ep in elbow_points_1based if ep > 5000]
            
            print(f"  Small scale ({method}): {small_range}")
            print(f"  Medium scale ({method}): {medium_range}")
            print(f"  Large scale ({method}): {large_range}")
            
            # Suggest practical ranges for experiments
            if len(elbow_points_1based) >= 3:
                practical_range = sorted(elbow_points_1based[:5])  # Top 5 elbow points
                print(f"  Recommended experimental range: {practical_range}")
        
        # Save summary to file
        summary_file = os.path.join(outdir, f"elbow_analysis_summary_{method}_RS{Random_state}_Fold{Fold}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Feature Selection Elbow Point Analysis Summary - {method.upper()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{method.upper()} Method:\n")
            f.write(f"Detected elbow points: {elbow_points_1based}\n\n")
            
            if elbow_points_1based:
                f.write("Detailed Analysis:\n")
                normalized_scores = sorted_scores / np.sum(sorted_scores)
                cumulative = np.cumsum(normalized_scores)
                
                for i, ep_1based in enumerate(elbow_points_1based):
                    ep = ep_1based - 1  # Convert back to 0-based
                    if ep < len(sorted_scores):
                        f.write(f"\nElbow Point {i+1}: {ep_1based} features\n")
                        f.write(f"  Score: {sorted_scores[ep]:.6f}\n")
                        f.write(f"  Cumulative importance: {cumulative[ep]*100:.2f}%\n")
        
        print(f"\nAnalysis complete! Plots saved to: {outdir}")
        print(f"Summary saved to: {summary_file}")
        
        return results


    def set_evaluation_function(self, eval_function = 'AccuracyEvaluation'):
        if eval_function == 'AccuracyEvaluation':
            self.eval_function = AccuracyEvaluation(self.estimator)
        elif eval_function == 'Fairness_Evaluation':
            self.eval_function = Fairness_Evaluation(self.estimator)
        else:
            raise ValueError("Unsupported evaluation function")


    def access_data(self,X_train, index):
        # print("the index is : ", index)
        if isinstance(X_train, np.ndarray):
            return X_train[:,index]
        elif isinstance(X_train, pd.Series):
            return X_train.iloc[:,index]
        elif isinstance(X_train, pd.DataFrame):
            return X_train.iloc[:,index]
        else:
            raise ValueError("Unsupported type for X_train")
    def _check_output(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(X, pd.Series):
            return X
        elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            return X.iloc[:, 0]
        else:
            return X

    def find_optimal_features(self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray], 
                              method: str = 'cv', max_features: int = None) -> List[str]:
        X, y = self._check_input(X, y)
        
        if method == 'cv':
            return self._cv_feature_selection(X, y, max_features)
        elif method == 'elbow':
            return self._elbow_feature_selection(X, y, max_features)
        else:
            raise ValueError("Method must be either 'cv' or 'elbow'")

    def _cv_feature_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], max_features: int = None) -> List[str]:
        features = list(X.columns)
        max_features = max_features or len(features)
        best_score = float('-inf')
        best_feature_set = []

        for i in range(1, max_features + 1):
            for feature_set in combinations(features, i):
                score = self.eval_function.evaluate(X.loc[:, list(feature_set)], y)
                if score > best_score:
                    best_score = score
                    best_feature_set = list(feature_set)

        self.best_features = best_feature_set
        self.best_score = best_score
        return best_feature_set

    def chi_squared_elbow_plot(self, X_train, y_train, X_test, y_test, max_features=None,selected_columns=None):
        """
        Perform Chi-squared feature selection and generate an elbow plot.
        
        Parameters:
        - X_train, y_train: Training data
        - X_test, y_test: Test data
        - max_features: Maximum number of features to consider. If None, use all features.
        
        Returns:
        - fig: The matplotlib figure object
        - selected_features: List of selected feature names
        - chi2_scores: Chi-squared scores for all features
        """
        if max_features is None:
            max_features = X_train.shape[1]
        
        # Calculate Chi-squared scores for all features
        selector = SelectKBest(chi2, k='all')
        selector.fit(X_train, y_train)
        chi2_scores = selector.scores_
        
        # Sort features by score
        feature_scores = list(zip(X_train.columns, chi2_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get cumulative scores
        cumulative_scores = np.cumsum([score for _, score in feature_scores])
        
        # Create elbow plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(cumulative_scores) + 1), cumulative_scores, marker='o')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cumulative Chi-squared Score')
        ax.set_title('Elbow Plot of Chi-squared Feature Selection')
        
        # Add vertical line at the elbow point (you may need to adjust this heuristic)
        elbow_point = np.argmax(np.diff(np.diff(cumulative_scores))) + 1
        ax.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow Point: {elbow_point}')
        
        ax.legend()
        plt.tight_layout()
        # save
        plt.savefig('Chi2_elbow_plot.png')
        
        # Select features based on the elbow point
        selected_features = [feature for feature, _ in feature_scores[:elbow_point]]
        
        return fig, selected_features, chi2_scores

    def genetic_algorithm(self, X, y, selected_columns,
                          population_size: int = 500, generations: int = 50, 
                          crossover_prob: float = 0.75, mutation_prob: float = 0.25) -> List[str]:
        self.GA_params['population_size'] = population_size
        self.GA_params['generations'] = generations
        self.GA_params['crossover_prob'] = crossover_prob
        self.GA_params['mutation_prob'] = mutation_prob
        # X, y = self._check_input(X, y)
        print("the X data type is : ", type(X))
        # print the parameters
        print('population_size:',population_size,'\ngenerations:',generations,'\ncrossover_prob:',crossover_prob,'\nmutation_prob:',mutation_prob)
        
        # Create types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        def individual():
            return [bool(np.random.randint(2)) for _ in range(X.shape[1])]   
        # Initialize toolbox

        # Define evaluation function
        def evalFeatures(individual):
            selected_features = [i for i, x in enumerate(individual) if x]
            if not selected_features:
                return 0,  # Return a tuple
            X_selected = self.access_data(X,selected_features)
            return (self.eval_function.evaluate(X_selected, y),)  # Return a tuple
        toolbox = base.Toolbox()
        # toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initIterate, creator.Individual, individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # Register genetic operators
        toolbox.register("evaluate", evalFeatures)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create initial population
        population = toolbox.population(n=population_size)
        hof = tools.HallOfFame(5)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)
        stats.register('std',np.std)
        # Run genetic algorithm
        population,logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, 
                            ngen=generations,halloffame = hof, verbose=False)

        # Select best individual
        gen = logbook.select("gen")
        avg_fitness = logbook.select("avg")
        std = logbook.select('std')

        best_individual = hof[0]
      
        selected_features = [i for i, x in enumerate(best_individual) if x]
        print('Selected features:', [i for i, x in enumerate(best_individual) if x])
        selected_features_name = [selected_columns[i] for i in selected_features]
        self.best_features = selected_features_name
        self.best_score = best_individual.fitness.values[0]
        print('Best individual:', best_individual)
        print('Best score:', best_individual.fitness.values[0])

        return self.best_features


    def Chi2(self, X_train, y_train, X_test, y_test, subsetsize):
        '''Perform Chi-squared feature selection and return selected features.
        Parameters:
        - X_train: Training feature set
        - y_train: Training labels
        - X_test: Test feature set
        - y_test: Test labels
        - subsetsize: Number of features to select
        Returns:
        - X_chi2_train(dataframe): Transformed training feature set with selected features
        - X_chi2_test(dataframe): Transformed test feature set with selected features
        - selected_columns_CHi: List of selected feature names
        - chi2_scores: Chi-squared scores for all features'''
        selector = SelectKBest(chi2, k=subsetsize)
        selector.fit(X_train, y_train)
        cols_chi = selector.get_support(indices=True)
        
        X_chi2_train = X_train.iloc[:, cols_chi]
        X_chi2_test = X_test.iloc[:, cols_chi]
        selected_columns_CHi = X_test.columns[cols_chi].tolist()
        # print("x_chi2_train",X_chi2_train)
        # print("x_chi2_test",X_chi2_test)
        return X_chi2_train, X_chi2_test, selected_columns_CHi, selector.scores_

    def Relief(self,X_train, y_train, X_test, y_test, subsetsize):
        fs = relief.Relief(n_features=subsetsize, random_state=3)
        Re_train, feat_train = fs.fit_transform(X_train.values, y_train.values)
        Re_test, feat_indices = fs.transform(X_test.values)
        feat_in = pd.Series(feat_indices)
        selected_columns = X_test.iloc[:, feat_indices].columns.tolist()
        re_scores = 0
        return Re_train, Re_test, selected_columns,re_scores

    def CMIM(self,X_train, y_train, X_test, y_test, subsetsize):

        # J_CMIM = scores
        selected_features_index, J_CMIM, MIfy = CMIM.cmim(
            X_train, y_train, n_selected_features=subsetsize)
        selected_features = X_test.iloc[:,
                                        selected_features_index].columns.tolist()
        selected_columns_CMIM = X_test.loc[:, selected_features].columns.tolist()
        X_cmim_train = X_train.loc[:, selected_features]
        X_cmim_test = X_test.loc[:, selected_features]
        print("JCMIM",J_CMIM)
        print("MIFY",MIfy)
        
        return X_cmim_train, X_cmim_test, selected_columns_CMIM ,J_CMIM

    def MI(self,X_train, y_train, X_test, y_test, subsetsize):
        
        X_new_MI = SelectKBest(mutual_info_classif, k=subsetsize)
        MI_train = X_new_MI.fit_transform(X_train, y_train)
        MI_test = X_new_MI.transform(X_test)
        # get the columns index and then use it to do the permutation test
        cols = X_new_MI.get_support(indices=True)
        MI_scores = X_new_MI.scores_
        selected_columns = X_test.iloc[:, cols].columns.tolist()
        return MI_train, MI_test, selected_columns,MI_scores
    
    def Relief_f(self,X_train, y_train, X_test, y_test, subsetsize):
        fs = relief.ReliefF(n_features=subsetsize, random_state=3)
        Re_train, feat_train = fs.fit_transform(X_train.values, y_train.values)
        Re_test, feat_indices = fs.transform(X_test.values)
        feat_in = pd.Series(feat_indices)
        selected_columns = X_test.iloc[:, feat_indices].columns.tolist()
        re_scores = 0
        return Re_train, Re_test, selected_columns,re_scores
   
