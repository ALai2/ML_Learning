import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.cluster import KMeans

import datasets as dt

# generate data
data_df = dt.data_frame_make_circles()

# use k-means to select clusters
k_means = KMeans(random_state=25, n_clusters=2)
k_means.fit(data_df)
cluster = k_means.predict(data_df)

# label points with cluster label
cluster = ['c_' + str(c) for c in cluster]

# plot points colored by cluster label
fig, ax = plt.subplots()
sns.scatterplot(x='x', y='y', data=data_df.assign(cluster = cluster), hue='cluster', ax=ax)
ax.set(title='K-Means Clustering')
plt.show()