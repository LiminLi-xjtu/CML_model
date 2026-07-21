import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from model import TCML
from utiles import dataset_params, load_dataset


if __name__ == '__main__':
    dataset = 'cora'  # You can select "cora", "citeseer", "wiki", "acm", "dblp", "cora_sorted" or "citeseer_sorted"
    X, A, gnd = load_dataset(dataset)
    params = dataset_params[dataset]
    print('Run dataset "{}"'.format(dataset))

    # run T-CML modal
    Z, acc, nmi, F1 = TCML(
        A,
        X,
        gnd,
        eta=params["eta"],
        alpha=params["alpha"],
        beta=params["beta"],
        lamb=params["lamb"],
        t=params["t"],
        max_iter=params["max_iter"],
    )

