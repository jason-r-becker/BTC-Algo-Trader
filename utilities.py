import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from tabulate import tabulate

plt.style.use('ggplot')


# print dataframes with nice aesthetic to console
def prettyPrint(dataframe):
    print(tabulate(dataframe, headers='keys', tablefmt='psql'))


# Plot histograms
def hists(data, bins=10, size='auto', c='steelblue', ctop=False, fs=12):
    if size == 'auto':
        cols = int(np.ceil(np.sqrt(len(list(data)))))  # optimal number of columns for square matrix of histograms
        rows = int(np.ceil(len(list(data)) / cols))  # otpimal number of rows for square matrix of histograms
    else:
        cols = size[0]
        rows = size[1]
    print('{} cols and {} rows '.format(cols, rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    i = 1
    for ax, feature in zip(axes.flatten(), list(data)):
        if ctop & (i <= len(list(data)) / 2):
            col = 'firebrick'
        else:
            col = c
        ax.hist(data[feature], bins=bins, color=col, alpha=0.8)
        ax.set_title("{}".format(feature.upper()), fontsize=fs)
        i += 1
    fig.subplots_adjust(hspace=0.4)
    plt.tight_layout()



# Show outliers
def outliers(data, show=False, save=False):
    outliers_i = []
    for feature in list(data):
        Q1 = np.percentile(data[feature], 25)
        Q3 = np.percentile(data[feature], 75)
        step = 1.5 * (Q3 - Q1)
        outliers_i += data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))].index.tolist()

        # Display the outliers
        if show:
            print("Data points considered outliers for the feature '{}':".format(feature))
            prettyPrint(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])

    if save:
        return np.unique(outliers_i)


# Correlation matrix of dataframe
def correlation(data, col='return', exclude=None, values=False, show=False, save=False, ret_col=False):
    if exclude is not None:
        data.drop(exclude, axis=1, inplace=True)

    corr = data.corr().abs()



    if values:
        print(corr.sort_values([col], ascending=False)[col][1:])

    if save:
        plt.savefig('{}correlation_plot.png'.format(path), bbox_inches='tight', format='png')

    if show:
        # Plot results
        fig, ax = plt.subplots(figsize=(len(corr.columns), len(corr.columns)))
        ax.matshow(corr)
        cmap = cm.get_cmap('coolwarm', 300)
        cax = ax.imshow(corr, interpolation="nearest", cmap=cmap)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        cax.set_clim([0, 1])
        fig.colorbar(cax, ticks=[0, .25, .5, .75, 1])
        plt.show()

    if ret_col:
        return corr.sort_values([col], ascending=False)[col][1:]

# PCA
def apply_pca(data, num_components, show=False, save=False):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components, random_state=0)
    pca = pca.fit(data)

    if save:
        reduced_data = pca.transform(data)
        cols = []
        for i in range(num_components):
            cols.append('dim {}'.format(i + 1))
        reduced_data = pd.DataFrame(reduced_data, columns=cols)
        if show:
            pca_results = vs.pca_results(data, pca)
            pca_results.cumsum()
            plt.savefig('{}pca_results.png'.format(path), bbox_inches='tight', format='png')
            plt.show()
        return reduced_data


# clusters
def cluster(data, solver='KNN'):
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    n_list = []
    score_list = []
    for n in range(2, 15):
        if solver == 'KNN':
            clusterer = KMeans(n_clusters=n, random_state=0, init='k-means++', )
        elif solver == 'GMM':
            clusterer = GaussianMixture(n_components=n, random_state=0)
        clusterer.fit(data)

        preds = clusterer.predict(data)
        score = silhouette_score(data, preds, random_state=0)
        n_list.append(n)
        score_list.append(score)

    plt.plot(n_list, score_list, '-o', color='steelblue')
    plt.title('Optimizing Number of Clusters')
    plt.ylabel('Silhoutte score')
    plt.xlabel('Number of Clusters')
    plt.show()

    optimal = np.argmax(score_list) + 2
    clusterer = GaussianMixture(n_components=optimal, random_state=0)
    clusterer.fit(data)
    preds = clusterer.predict(data)
    centers = clusterer.means_

    # plot results of the clustering
    vs.cluster_results(data, preds, centers)


# pair plots
def pair_plots(data):
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(data)
    plt.show()


# corr_df = correlation(df, save=False, show=True)
# outliers_array = outliers(df, show=True, save=True)
# hists(df, 20)
# log_df = np.log(df)
# hists(log_df, 3, 2)
# pca_df = apply_pca(clust_df, 3, show=True, save=True)
# pca_df.to_csv('listings_pca_reduced.csv')
# prettyPrint(pca_df.head())
# pca_df_sample = pca_df.sample(frac=0.10)
# cluster(pca_df_sample, solver='KNN')

# pair_plots(df)
