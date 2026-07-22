import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from model import TCML
from utiles import load_and_process_data, stage_dataset_params

if __name__ == '__main__':
    dataset = 'citeseer'  # 'cora', 'citeseer' or 'acm'
    stage = 3   # stage>=2
    A, X, gnd = load_and_process_data(dataset, stage)
    params = stage_dataset_params[dataset][stage]

    Z, acc, nmi, F1 = TCML(
        A, X, gnd,
        eta=params["eta"],
        alpha=params["alpha"],
        beta=params["beta"],
        lamb=params["lamb"],
        t=params["t"],
        max_iter=params["max_iter"])
