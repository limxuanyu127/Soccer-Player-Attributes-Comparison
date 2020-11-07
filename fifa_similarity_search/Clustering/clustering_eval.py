import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def silhouette_blob(samples, cluster_labels, cluster_centres=None, title=None, save_link=None):
    n_clusters = len(np.unique(cluster_labels))
    
    # Create a subplot with 1 row and 2 columns
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 10)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(samples) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(samples, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(samples, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        '''
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(samples[:, 0], samples[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        if cluster_centres is not None:
            # Draw white circles at cluster centers
            ax2.scatter(cluster_centres[:, 0], cluster_centres[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(cluster_centres):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        '''

        plt.suptitle(("Silhouette analysis for %s" % (title)),
                     fontsize=14, fontweight='bold')
    
    if save_link:
        plt.savefig('{}/{} Silhouette Scores.png'.format(save_link, title))
    plt.show()

    
# Method needs to be changed/edited    
def labels_in_cluster(given_cluster, num_clusters, title=None):
    y_train_vals = y_train.values
    classes_in_cluster = np.zeros(shape=(num_clusters, len(labels)), dtype=int)
    for i in range(num_clusters):
        dataInd = np.argwhere(given_cluster==i).flatten()
        for ind in dataInd:
            if ',' in y_train_vals[ind]:
                split = y_train_vals[ind].split(', ')
                for s in split:
                    j, = np.where(labels == s)
                    classes_in_cluster[i][j] += 1
            else:
                j, = np.where(labels == y_train_vals[ind])
                classes_in_cluster[i][j] += 1

    plt.figure(figsize=(15,5))
    if title:
        plt.title("Labels within clusters by {}".format(title))
    else:
        plt.title("Labels within clusters")
    sns.heatmap(classes_in_cluster, annot=True, cmap='Blues', fmt="d")
    plt.xticks([(i+0.5) for i in np.arange(len(labels))], labels=labels)
    plt.show()
    
    
def cosine_matrix(samples, labels, title=None, save_link=None):
    unique_labels, counts = np.unique(labels, return_counts=True)
    tick_loc = [(sum(counts[:i])+counts[i]/2) for i in np.arange(len(unique_labels))]
    
    num_samples = samples.shape[0]
    sortedInd = np.argsort(labels)
    
    plt.figure(figsize=(20,15))
    if title:
        plt.title("Cosine Matrix by Clusters from {}".format(title))
    else:
        plt.title("Cosine Matrix by Clusters")
    sns.heatmap(cosine_similarity(samples[sortedInd]), cmap='Blues')
    plt.yticks(tick_loc, labels=unique_labels)
    plt.xticks(tick_loc, labels=unique_labels, rotation='horizontal')
    
    if save_link:
        plt.savefig('{}/{} Cosine Similarity.png'.format(save_link, title))
    plt.show()