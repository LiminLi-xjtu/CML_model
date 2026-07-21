from model import ACML
from utiles import clustering_metrics, update_local_best, spectral_clusteringA_mean, kmeans_clustering_mean, ACML_simulation_params





if __name__ == '__main__':
    import scipy.io as sio
    mat_path = 'data/graph_noise_epsilon_square2.mat'
    mat_data = sio.loadmat(mat_path)

    A_name = 'A0_1'
    X_name = 'X0_7'
    A = mat_data[A_name]
    X = mat_data[X_name].T
    gnd = mat_data['labels'].flatten()
    k = 3
    A_val = A_name[1:].replace('_', '.')
    X_val = X_name[1:].replace('_', '.')
    key = f"A{A_val}+X{X_val}"
    acml_params = ACML_simulation_params[key]

    # Run ACML
    Z, C, acc, nmi, f1 = ACML(A, X, gnd, **acml_params)
    ac, _, nm, _, f1, _ = spectral_clusteringA_mean(Z, gnd, n_iter=2, k=k)
    print(f'ACML: ACC={ac:.4f}, NMI={nm:.4f}, F1={f1:.4f}')


    # X_n = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    # acc_mean, acc_std, nmi_mean, nmi_std, f1_mean, f1_std = kmeans_clustering_mean(X_n, gnd, n_iter=100, k=3)
    # print('KMEANS', 'acc:{:.4f}'.format(acc_mean), 'nmi:{:.4f}'.format(nmi_mean), 'f1:{:.4f}'.format(f1_mean))
    # ac, _, nm, _, f1, _ = spectral_clusteringA_mean(A, gnd, n_iter=50, k=3)
    # print('spectral', 'acc:{:.4f}'.format(ac), 'nmi:{:.4f}'.format(nm), 'f1:{:.4f}'.format(f1))


