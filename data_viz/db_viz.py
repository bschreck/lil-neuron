import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
client = MongoClient()
db = client['lil-neuron-db']


def gen_hist(attr='popularity', use_log_axis=False, bins=50, max_x=None):
    artists = db.artists.find()
    attrs = {}
    for artist in artists:
        if artist[attr]:
            attrs[artist['name']] = artist[attr]
    # the histogram of the data
    np_attrs = np.array(attrs.values())
    _min = np_attrs.min()
    _max = np_attrs.max()

    n, bins, patches = plt.hist(np_attrs, bins=bins, normed=1, facecolor='green', alpha=0.75)
    if use_log_axis:
        plt.gca().set_xscale("log")
    plt.xlabel(attr)
    plt.ylabel('Probability')
    plt.title('Histogram of Rapper {}'.format(attr[0].upper() + attr[1:]))
    y_max = n.max()
    if max_x is None:
        max_x = _max
    plt.axis([_min, max_x, 0, y_max * 2])
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    gen_hist('followers', use_log_axis=True)
