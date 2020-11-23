import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from itertools import chain
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

# Set random state. 
rs = np.random.seed(25)

# generate one circle data with random Gaussian noise
def generate_circle_sample_data(r, n, sigma):
    angles = np.random.uniform(low=0, high=2*np.pi, size=n)

    x_epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)
    y_epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)

    x = r*np.cos(angles) + x_epsilon
    y = r*np.sin(angles) + y_epsilon
    return x, y

# generates multiple circles data with random Gaussian noise
def generate_concentric_circles_data(param_list):
    coordinates = [ 
        generate_circle_sample_data(param[0], param[1], param[2])
     for param in param_list
    ]
    return coordinates

# return coordinates of concentric circles as dataframe
def data_frame_concentric_circles(): 
    n = 1000                # number of points per circle.
    r_list =[2, 4, 6, 8]    # radius of concentric circles
    sigma = 0.25            # standar deviation (Gaussian noise)

    param_list = [(r, n, sigma) for r in r_list]    

    coordinates = generate_concentric_circles_data(param_list)
    xs = chain(*[c[0] for c in coordinates])
    ys = chain(*[c[1] for c in coordinates])

    return pd.DataFrame(data={'x': xs, 'y': ys})

# return coordinates of two moons as dataframe
def data_frame_from_moons():
    X, y = make_moons(n_samples=1000, noise=.1)
    return pd.DataFrame(data={'x': X.T[0], 'y': X.T[1]})

# return coordinates of two concentric circles as dataframe
def data_frame_make_circles():
    X, y = make_circles(n_samples=1000, noise=.05, factor=0.5)
    return pd.DataFrame(data={'x': X.T[0], 'y': X.T[1]})

# Plot the input data of dataset
def plot_input_data(data_df):
    fig, ax = plt.subplots()
    sns.scatterplot(x='x', y='y', color='black', data=data_df, ax=ax)
    ax.set(title='Input Data')
    plt.show()