from sklearn import metrics
from munkres import Munkres
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore", message="Graph is not fully connected", category=UserWarning)
warnings.filterwarnings("ignore", message="Array is not symmetric")
# import sys
# sys.path.append('/home/wangxi/Attribute_graph_clustering/my-method')



def filter_features_sparse(A, X, t):
    """
    Performs a graph filtering operation on the feature matrix X, computed using a sparse matrix.

    Parameters.
        A (numpy.ndarray or scipy.sparse): the adjacency matrix (n x n).
        X (numpy.ndarray): feature matrix (n x d). t (int): filter matrix.
        t (int): the time step of the filter, which affects the filter strength.

    Returns.
        numpy.ndarray: the filtered eigenmatrix X.
    """

    if not sp.issparse(A):
        A = sp.csr_matrix(A)

    n = A.shape[0]
    I = np.eye(n)
    A = A + I
    D = np.sum(A, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D = np.diagflat(D)
    D_sparse = csr_matrix(D)
    A_sparse = csr_matrix(A)
    A = A_sparse.dot(D)
    A = D_sparse.dot(A)

    # Get filter G
    Ls = I - A
    G = I - 0.5 * Ls
    G_sparse = csr_matrix(G)

    for i in range(t):
        X = G_sparse.dot(X)
    if issparse(X):
        X = X.toarray()
    return X


def spectral_clusteringA_mean(A, gnd, n_iter=50, k=3):
    """
    Performs multiple clustering using spectral clustering and outputs mean and variance.

    Parameters.
    - A: precomputed adjacency matrix
    - gnd: true labels
    - n_iter: number of iterations for clustering, default 50
    - k: number of clusters, default 3

    Returns: None, output the result directly.
    - None, output the result directly
    """

    accuracies = []
    nmi_scores = []
    f1_scores = []

    for random_state in range(n_iter):
        clustering = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=random_state,
                                        affinity='precomputed').fit(A)
        # A + np.eye(A.shape[0]) * 1e-5
        y_A = clustering.labels_

        cm = clustering_metrics(gnd, y_A)
        ac, nm, f1 = cm.evaluationClusterModelFromLabel()

        accuracies.append(ac)
        nmi_scores.append(nm)
        f1_scores.append(f1)

    mean_ac = np.mean(accuracies)
    std_ac = np.std(accuracies)

    mean_nm = np.mean(nmi_scores)
    std_nm = np.std(nmi_scores)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    return mean_ac, std_ac, mean_nm, std_nm, mean_f1, std_f1
# Usage Example:
# acc_mean, acc_std, nmi_mean, nmi_std, f1_mean, f1_std = spectral_clustering_mean(A, gnd, n_iter=100, k=3)


def spectral_clusteringX_mean(X, gnd, n_iter=50, k=3):
    """
    Performs multiple clustering using spectral clustering and outputs the mean and variance.

    Parameters.
    - A: precomputed adjacency matrix
    - gnd: true labels
    - n_iter: number of iterations for clustering, default 50
    - k: number of clusters, default 3

    Returns: None, output the result directly.
    - None, output the result directly
    """

    accuracies = []
    nmi_scores = []
    f1_scores = []

    for random_state in range(n_iter):
        clustering = SpectralClustering(n_clusters=k, assign_labels="discretize",random_state=random_state).fit(X)
        y_A = clustering.labels_

        cm = clustering_metrics(gnd, y_A)
        ac, nm, f1 = cm.evaluationClusterModelFromLabel()

        accuracies.append(ac)
        nmi_scores.append(nm)
        f1_scores.append(f1)

    mean_ac = np.mean(accuracies)
    std_ac = np.std(accuracies)

    mean_nm = np.mean(nmi_scores)
    std_nm = np.std(nmi_scores)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    # print(f'Spectral Clustering X results (mean and std over {n_iter} iterations):')
    # print(f'Accuracy Mean: {mean_ac}, Std: {std_ac}')
    # print(f'NMI Mean: {mean_nm}, Std: {std_nm}')
    # print(f'F1 Score Mean: {mean_f1}, Std: {std_f1}\n')
    return mean_ac, std_ac, mean_nm, std_nm, mean_f1, std_f1
# Usage Example:
# acc_mean, acc_std, nmi_mean, nmi_std, f1_mean, f1_std = spectral_clustering_mean(A, gnd, n_iter=100, k=3)



def spectral_clustering(Z_sym, k, random_state=23):
    u, _, _ = sp.linalg.svds(Z_sym, k=k, which='LM')
    u = u.astype(float)
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(u)
    predict_labels = kmeans.predict(u)
    return predict_labels


def kmeans_clustering_mean(u, gnd, n_iter=50, k=3):
    """
    Performs multiple clustering using KMeans and returns the mean and variance.

    Parameters.
    - u: data matrix (usually a feature matrix or a reduced dimension matrix)
    - gnd: true label
    - n_iter: number of iterations for clustering, default 50
    - k: number of clusters, default 3

    Returns.
    - acc_mean: the mean of the accuracy.
    - acc_std: standard deviation of the accuracy.
    - nmi_mean: mean value of NMI
    - nmi_std: standard deviation of NMI
    - f1_mean: Mean of F1 Score
    - f1_std: standard deviation of F1 Score
    """

    accuracies = []
    nmi_scores = []
    f1_scores = []

    for random_state in range(n_iter):
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(u)
        predict_labels = kmeans.predict(u)

        cm = clustering_metrics(gnd, predict_labels)
        ac, nm, f1 = cm.evaluationClusterModelFromLabel()

        accuracies.append(ac)
        nmi_scores.append(nm)
        f1_scores.append(f1)

    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)

    nmi_mean = np.mean(nmi_scores)
    nmi_std = np.std(nmi_scores)

    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    # print(f'Kmeans Clustering results (mean and std over {n_iter} iterations):')
    # print(f'Accuracy Mean: {acc_mean}, Std: {acc_std}')
    # print(f'NMI Mean: {nmi_mean}, Std: {nmi_std}')
    # print(f'F1 Score Mean: {f1_mean}, Std: {f1_std}\n')
    return acc_mean, acc_std, nmi_mean, nmi_std, f1_mean, f1_std
# Usage Example:
# acc_mean, acc_std, nmi_mean, nmi_std, f1_mean, f1_std = kmeans_clustering_mean(u, gnd, n_iter=50, k=3)


def gaussian_kernel_sigma(X, sigma):
    dists = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
    # sigma = np.mean(dists)
    dists = dists.astype(np.float64)
    sigma = np.float64(sigma)
    K = np.exp(-sigma*dists)
    return K

def gaussian_kernel(X):
    dists = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
    sigma = np.mean(dists)
    dists = dists.astype(np.float64)
    sigma = np.float64(sigma)
    K = np.exp(-sigma*dists)  #K = np.exp(- dists/(2*sigma**2))
    return K


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro


    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)  # Add ARI calculation
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        # print('ACC=%f, f1_macro=%f, precision_macro=%f, ARI=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, ari, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f,ARI=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, ari, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, f1_macro




def update_local_best(err, err_old, acc, nmi, F1, iter, local_best_acc, local_best_nmi, local_best_f1,
                      local_best_iter, max_iter, tau):
    if err < 0.05 and (acc[iter - 1] + nmi[iter - 1] + F1[iter - 1] > local_best_acc + local_best_nmi + local_best_f1):
        local_best_acc = acc[iter - 1]
        local_best_nmi = nmi[iter - 1]
        local_best_f1 = F1[iter - 1]
        local_best_iter = iter - 1
    if err_old < err and (
            acc[iter - 1] + nmi[iter - 1] + F1[iter - 1] > local_best_acc + local_best_nmi + local_best_f1):
        local_best_acc = acc[iter - 1]
        local_best_nmi = nmi[iter - 1]
        local_best_f1 = F1[iter - 1]
        local_best_iter = iter - 1
    return local_best_acc, local_best_nmi, local_best_f1, local_best_iter



dataset_params = {
    "cora": {"eta": 300000, "alpha": 115, "beta": 1, "lamb": 0.6, "t": 16, "max_iter": 50, },
    "citeseer": {"eta": 600000, "alpha": 42, "beta": 0.01, "lamb": 0, "t": 8, "max_iter": 50, },
    "wiki": {"eta": 300000, "alpha": 150, "beta": 1, "lamb": 3, "t": 4, "max_iter": 50, },  #"eta": 600000, "alpha": 1, "beta": 1, "lamb": 3, "t": 4, "max_iter": 100
    "acm": {"eta": 2800000, "alpha": 200, "beta": 1, "lamb": 0, "t": 5, "max_iter": 30, },
    "dblp": {"eta": 1000000, "alpha": 50, "beta": 0.5, "lamb": 0.1, "t": 1, "max_iter": 30, },
    "cora_sorted": {"eta": 300000, "alpha": 115, "beta": 1, "lamb": 0.5, "t": 16, "max_iter": 30, },
    "citeseer_sorted": {"eta": 600000, "alpha": 42, "beta": 0.01, "lamb": 0, "t": 8, "max_iter": 50, },
}


def load_dataset(dataset):
    """
     Parameters.
    - dataset: dataset name (“cora”, “citeseer”, “wiki”, “acm”, “dblp”, “cora_sorted”, “citeseer_sorted”)

     Returns.
    - X: Feature matrix
    - A: adjacency matrix
    - gnd: label vector
    """

    if dataset in ['cora_sorted', 'citeseer_sorted']:
        features_struct = np.load(f'data/{dataset}.npz')
        X = features_struct['features']
        gnd = features_struct['labels'] - 1  # 标签从 0 开始
        A = features_struct['adjacency']
    elif dataset in ["cora", "citeseer", "wiki", "acm", "dblp","pubmed"]:
        data = sio.loadmat(f"data/{dataset}.mat")
        X = data["fea"]
        A = data["W"]
        gnd = data["gnd"]

        if dataset in ["cora", "citeseer", "wiki","pubmed"]:
            gnd = gnd.T[0] - 1
        elif dataset in ["acm", "dblp"]:
            gnd = gnd[0, :]
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    return X, A, gnd



stage_dataset_params = {
    "cora": {
        2: {"eta": 500000, "alpha": 100, "beta": 1, "lamb": 0, "t": 4, "tau": 0.005, "max_iter": 50, "k": 7},
        3: {"eta": 650000, "alpha": 100, "beta": 1, "lamb": 0, "t": 4, "tau": 0.005, "max_iter": 50, "k": 7},
    },
    "citeseer": {
        2: {"eta": 600000, "alpha": 4, "beta": 1, "lamb": 0.4, "t": 4, "tau": 0.005, "max_iter": 30, "k": 6},
        3: {"eta": 600000, "alpha": 4, "beta": 1, "lamb": 0.4, "t": 4, "tau": 0.005, "max_iter": 30, "k": 6},
    },
    "acm": {
        2: {"eta": 2800000, "alpha": 200, "beta": 1, "lamb": 0, "t": 1, "tau": 0.05, "max_iter": 10, "k": 3},
        3: {"eta": 2800000, "alpha": 200, "beta": 1, "lamb": 0, "t": 1, "tau": 0.05, "max_iter": 100, "k": 3},
    },
}


def has_nonpositive_element(matrix):
    return np.any(matrix <= 0)

def replace_negative_with_zero(matrix):
    matrix[matrix < 0] = 0
    return matrix

def load_and_process_data(dataset, stage):
    if dataset in ["cora", "citeseer"]:
        features_struct = np.load(f'data/{dataset}_sorted.npz')
        X = features_struct['features']
        gnd = features_struct['labels'] - 1
        adj_file = f"data/{dataset}Z{stage-1}.npz"
    else:
        data = sio.loadmat(f"data/{dataset}.mat")
        X = data["fea"]
        gnd = data["gnd"][0, :]
        adj_file = f"data/{dataset}Z{stage-1}.npz"

    A = np.load(adj_file)['adjacency']
    if has_nonpositive_element(A):
        A = replace_negative_with_zero(A)
    return A, X, gnd




ACML_simulation_params = {
"A0.1+X0.1":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.1+X0.3":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.1+X0.5":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.1+X0.7":  {"eta": 10000, "K": 3, "alpha": 1, "beta": 1, "gamma": 10, "lamb": 1, "k_max": 250, "tol": 0.001 },
"A0.3+X0.1":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.3+X0.3":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.3+X0.5":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.3+X0.7":  {"eta": 10000, "K": 3, "alpha": 5, "beta": 1, "gamma": 50, "lamb": 1, "k_max": 250, "tol": 0.001 },
"A0.5+X0.1":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 500, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.5+X0.3":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 500, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.5+X0.5":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.5+X0.7":  {"eta": 10000, "K": 3, "alpha": 5, "beta": 1, "gamma": 50, "lamb": 1, "k_max": 250, "tol": 0.001 },
"A0.7+X0.1":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.7+X0.3":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.7+X0.5":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.7+X0.7":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 100, "lamb": 1, "k_max": 250, "tol": 0.001 },
"A0.9+X0.1":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 500, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.9+X0.3":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 500, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.9+X0.5":  {"eta": 10000, "K": 3, "alpha": 10, "beta": 1, "gamma": 500, "lamb": 1, "k_max": 150, "tol": 0.001 },
"A0.9+X0.7":  {"eta": 10000, "K": 3, "alpha": 3, "beta": 1, "gamma": 500, "lamb": 1, "k_max": 100, "tol": 0.001 },
}