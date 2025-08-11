
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class checkLG():
    def __init__(self, data, location, str):
        '''
        :location: location dataset to get chromosomes and locations
        :data: data set to help find the gene values
        :str: the name of the chromosome sequence we need
        '''
        self.data = data
        self.location = location
        self.str = str
        self.LG = self.location[self.location['Chromosome/scaffold'].str.contains(
            r'\b{}\b'.format(str))]
        self.snapperLG = self.get_the_LG()

    def get_the_LG(self):
        """ Get chromosomes with gene values
       
        Return: gene values with the same chromosome.
        """
        # let the snapper data match to the Chromosome
        # # Check if indices exist in the DataFrame's index using .isin()
        valid_indices = self.data.columns.isin(self.LG.Name.values)
        snapper_LG = self.data.loc[:, valid_indices]
        return snapper_LG

    # # silhouette,calinski_harabasz
    # def run_elbow(self, kmeans, metirc='distortion'):
    #     data = self.LG.Position.values.reshape(-1, 1)
    #     # Chromosomes come in different lengths, so make them dynamic
    #     min = int(len(data)/6)
    #     max = int(len(data)/2)
    #     print("the range of the K : ({},{})".format(min, max))
    #     visualizer = KElbowVisualizer(kmeans, k=(min, max), metric=metirc)
    #     visualizer.fit(data)
    #     visualizer.show()
    #     visualizer1 = KElbowVisualizer(kmeans, k=(
    #         min, max), metric='silhouette',  locate_elbow=False)
    #     visualizer1.fit(data)
    #     scores = visualizer1.k_scores_
    #     best_k_index = np.argmax(scores)
    #     best_k = visualizer1.k_values_[best_k_index]
    #     best_score = scores[best_k_index]
    #     print(f"Length: {len(data)}")
    #     print(f"Best K: {best_k}")
    #     ratio = len(data)/best_k
    #     print(f"Highest Score: {best_score}")
    #     print(f"There can probably be {ratio} genes in each clusters ")
    #     visualizer1.show()
    #     visualizer2 = KElbowVisualizer(kmeans, k=(
    #         min, max), metric='calinski_harabasz', locate_elbow=False)
    #     visualizer2.fit(data)
    #     visualizer2.show()
    #     return visualizer.elbow_value_, best_k, visualizer2.elbow_value_

    def run_Kmeans(self, best_K):
        kmeans = KMeans(n_clusters=best_K)
        kmeans.fit(self.LG.Position.values.reshape(-1, 1))
        # LG11
        # Get the cluster labels assigned to each data point
        labels = kmeans.labels_
        # Get the cluster centers
        centers = kmeans.cluster_centers_
        self.show_position(centers, labels)
        self.LG['Label'] = labels
        return self.LG

    # DBSCAN
    def run_DBSCAN(self, threshold):
        clustering = DBSCAN(eps=threshold, min_samples=1).fit(
            self.LG.Position.values.reshape(-1, 1))
        label = clustering.labels_
        # self.show_position(labels=label)
        self.LG['DBLabel'] = label
        return self.LG[(
            self.LG.Name.isin(self.snapperLG.columns))]

    def show_position(self, centers=None, labels=None):
        x = self.LG.Position
        plt.figure(figsize=(500, 20))
        y = [0] * len(x)  # A constant y-value of 0
        plt.scatter(x/10000, y, s=500, c=labels)
        if centers != None:
            centers_y = [0.0001]*len(centers)
            plt.scatter(centers/10000, centers_y, s=1000, marker='*', c='red')
        plt.yticks([])
        plt.xticks(x/10000, x, rotation=90)
        plt.ylim(-0.001, 0.01)
        plt.xticks(fontsize=45)
        plt.title("Show: {}".format(self.str), fontsize=40)
        plt.show()

    def show_Plot(self, bins=150):
        print(self.LG)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< {} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>".format(self.str))
        # print out the difference from the previous one
        print("Minimum difference :", self.LG.Position.diff().min())
        diff = self.LG.Position.diff()
        diff = diff.dropna()
        fig = plt.figure()  # an empty figure with no Axes
        # a figure with a 2x2 grid of Axes
        fig, axs = plt.subplots(3, 1, figsize=(20, 15))
        axs[0].boxplot(diff, vert=False)
        # Add labels to minimum, maximum, and quartile values
        axs[0].text(np.min(diff), 0.5, 'Min :{}'.format(
            np.min(diff)), ha='center', va='bottom', rotation=90)
        axs[0].text(np.percentile(diff, 25), 0.5, 'Q1 : {}'.format(
            np.percentile(diff, 25)), ha='center', va='bottom', rotation=90)
        axs[0].text(np.median(diff), 0.5, 'Median : {}'.format(
            np.median(diff)), ha='center', va='bottom', rotation=90)
        axs[0].text(np.percentile(diff, 75), 0.5, 'Q3 : {}'.format(
            np.percentile(diff, 75)), ha='center', va='bottom', rotation=90)
        axs[0].text(np.max(diff), 0.5, 'Max : {}'.format(
            np.max(diff)), ha='center', va='bottom', rotation=90)
        axs[0].set_title(
            "Box plots of positional differences on chromosomes : {}".format(self.str))
        print("One-quarter : ", np.percentile(diff, 25))
        print("Median : ", np.median(diff))
        print("Q3 : ", np.percentile(diff, 75))
        print("Max : ", np.max(diff))

        axs[1].bar(diff.index, diff)
        # Set the font size of x and y ticks
        axs[1].tick_params(axis='x', labelsize=13)
        axs[1].tick_params(axis='y', labelsize=13)
        # axs[1].text(0.5,np.percentile(diff, 25), 'Q1 : {}'.format(np.percentile(diff, 25)), ha='center', va='bottom')
        print(diff.sort_values(ascending=False).index[0])
        # axs[1].axvline(x =diff.sort_values(ascending = False ).index[0], color='red', linestyle='--' )
        # Only 90% of the values are maintained, as the differences are so great that it makes it difficult to display the graphs.
        diff_90_dataindex = self.LG.Position[self.LG.Position.diff(
        ) <= np.percentile(diff, 90)].index
        diff_90 = diff.loc[diff_90_dataindex]
        axs[2].hist(diff_90, bins)
        axs[2].set_title(
            "Frequency of difference values: Only 90% of the values are maintained ")
        plt.xticks(rotation=90)
        plt.locator_params(axis="x", nbins=bins)
        # this show the number of the instance
        count = len([num for num in diff if num <= np.percentile(diff, 20)])
        print("There are {} genes with differences below {}.".format(
            count, np.percentile(diff, 20)))
        plt.show()


# 'LG11', 'LG2', 'LG3', 'LG5', 'LG7', 'LG10', 'LG13', 'LG8', 'LG4', 'LG1',
#    'LG18', 'LG16', 'LG22', 'LG6', 'LG15', 'LG21', 'LG19', 'LG14', 'LG9',
#    'LG12', 'LG20', 'LG17', 'LG23', 'LG24', 'LG25', 'Super_scaffold_182'
