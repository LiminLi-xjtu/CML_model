import numpy as np  # Import the NumPy library for numerical computing
from scipy import linalg
from scipy.sparse import eye, issparse, triu  # Import the eye function from the SciPy library for creating sparse identity matrices
from scipy.sparse.linalg import eigs  # Import the eigs function from the SciPy library for computing eigenvalues and eigenvectors of sparse matrices
from utiles import clustering_metrics, update_local_best, filter_features_sparse, spectral_clustering, spectral_clusteringA_mean
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time



def TCML(A, X, gnd, eta, alpha, beta, lamb, t, max_iter=10):
    """
    Attributed Graph Refinement via low rank approximation and subspace learning (T-CML)
    """
    gamma = 0.001
    n = A.shape[0]
    k = len(np.unique(gnd))

    if isinstance(A, np.ndarray):
        Anorm = np.linalg.norm(A, ord=None)
    else:
        Anorm = np.linalg.norm(A.A, ord=None)

    scale = np.sqrt(k) / Anorm
    As = (A + A.T) * (scale / 2)
    print('1')

    # filter X
    I = np.eye(n)
    X = filter_features_sparse(A, X, t)
    if issparse(X):
        X = X.todense()

    Xnorm = np.linalg.norm(X, ord=None)
    scaleX = np.sqrt(k) / Xnorm
    X = X * scaleX
    alpha_XX = alpha * np.dot(X, X.T)
    print('2')

    Z = 0
    C = As
    acc = []
    nmi = []
    F1 = []

    y_A = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=23, affinity='precomputed').fit(As).labels_
    cm = clustering_metrics(gnd, y_A)
    ac, nm, f1 = cm.evaluationClusterModelFromLabel()
    acc.append(ac)

    local_best_acc = 0.0
    local_best_nmi = 0.0
    local_best_f1 = 0.0
    local_best_iter = 0

    err = 1
    iter = 0
    tau = 0.005

    while err > tau and iter < max_iter:  # While the error is greater than the convergence tolerance and the maximum number of iterations has not been reached,
        iter += 1
        Z_old = Z  # store the current value of Z
        err_old = err
        start_time = time.time()

        # Update U
        U = eigs(C, k=k, which='LM')[1]  # compute the eigenvectors of Z corresponding to its largest eigenvalues
        UU = np.real(np.dot(U, U.T))

        # Update Z
        AAA = (beta + 1 + gamma) * np.eye(n) + alpha_XX
        BBB = As + beta * UU + alpha_XX + gamma * Z_old - lamb * I
        C = linalg.solve(AAA, BBB)
        C = C.astype(float)

        D = C * (C > 0)

        # Update Z to T_eta(Z,D) by truncating Z:
        Z = shrink(C, D, eta, n)

        err = np.linalg.norm(Z - Z_old, ord=None)
        err = err / np.linalg.norm(Z, ord=None)

        end_time = time.time()
        print(f"Time spent in iteration {iter}: {round(end_time - start_time, 4)} s")

        Z_sym = (np.abs(Z) + np.abs(Z.T)) / 2
        predict_labels = spectral_clustering(Z_sym, k=k, random_state=23)
        cm = clustering_metrics(gnd, predict_labels)
        ac, nm, f1 = cm.evaluationClusterModelFromLabel()
        print(ac, nm, f1)
        acc.append(ac)
        nmi.append(nm)
        F1.append(f1)

        local_best_acc, local_best_nmi, local_best_f1, local_best_iter = update_local_best(
            err, err_old, acc, nmi, F1, iter,
            local_best_acc, local_best_nmi, local_best_f1, local_best_iter,
            max_iter, tau)
        if iter == max_iter or err < tau:
            print('local_best_acc: {:.4f}'.format(local_best_acc),
                  'local_best_nmi: {:.4f}'.format(local_best_nmi),
                  'local_best_f1: {:.4f}'.format(local_best_f1))
    return Z, acc, nmi, F1



def shrink(Z, D, eta, n):  # Define a helper function for truncating a matrix
    eta_1 = int(eta / 2)  # Compute half of eta rounded down to an integer
    D_upper = triu(D, k=1).tocoo()
    dd = D_upper.data
    dd.sort()  # Sort dd in ascending order
    dd = dd[::-1]  # Reverse dd to obtain it in descending order
    entry = ((eye(n) + (D >= dd[eta_1])) != 0)
    return Z * entry





def ACML(A, X, gnd, eta, K, alpha, beta, gamma, lamb,  k_max=10, tol=1e-3):
    n = A.shape[0]
    norm_A = np.linalg.norm(A, ord='fro')
    scale = np.sqrt(K) / (norm_A)
    As = (A + A.T) * (scale / 2.0)
    norm_X = np.linalg.norm(X, ord='fro')
    X = np.sqrt(K) / (norm_X) * X
    I = np.eye(n)


    if issparse(X):
        X = X.todense()
    alpha_XX = alpha * np.dot(X, X.T)

    # Initialization
    Z_k = np.zeros((n, n))
    C_k = np.zeros((n, n))
    s = 1


    err_list=[]
    acc_list = []
    nmi_list = []
    f1_list = []
    local_best_acc = 0.0
    local_best_nmi = 0.0
    local_best_f1 = 0.0
    local_best_iter = 0

    for k_iter in range(k_max):
        Z_old = Z_k.copy()
        C_prev = C_k.copy()


        # ----- UPDATE V -----
        C_sym = (C_k + C_k.T) / 2.0
        eigvals, V = eigs(C_sym, k=min(K, max(1, n - 1)), which='LM')
        V = np.real(V)
        VVT = np.real(np.dot(V, V.T))

        # ----- UPDATE C（USE beta, VVT, C_prev） -----
        C_k = solve_C(alpha_XX, I, beta, gamma, s, Z_k, lamb, VVT=VVT, C_prev=C_prev)
        # C_k = traceC(alpha_XX, I, beta, gamma, s, Z_k, lamb, VVT=VVT, C_prev=C_prev)

        # ----- UPDATE S_p -----
        S_p = (np.abs(C_k) + np.abs(C_k.T)) / 2.0

        # ----- UPDATE U-----
        eigvals, U = eigs(Z_k, k=K, which='LM')
        U = np.real(U)

        # ----- UPDATE Z-----
        Z_k = solve_Z_inner(As, U, S_p, beta, gamma, s, eta, n)

        # ----- UPDATE s  -----
        norm_Sp = np.linalg.norm(S_p, ord='fro')
        s = np.sqrt(K) / norm_Sp


        # ----- Convergence judgment, recording, and printing -----
        err = np.linalg.norm(Z_k - Z_old, ord='fro')
        err_list.append(err)

        try:
            ac, _, nm, _, f1, _ = spectral_clusteringA_mean(Z_k, gnd, n_iter=2, k=K)
        except Exception as e:
            print(f"[Warning] Evaluation failed at iteration {iter}: {e}")
            ac, nm, f1 = -1, -1, -1

        acc_list.append(ac)
        nmi_list.append(nm)
        f1_list.append(f1)
        print(f'Iter {k_iter + 1}: ACC={ac:.4f}, NMI={nm:.4f}, F1={f1:.4f}, err={err:.6f}')


        if k_iter == 0:
            err_old = err + 1
        else:
            err_old = np.linalg.norm(Z_k - Z_old, ord='fro')

        local_best_acc, local_best_nmi, local_best_f1, local_best_iter = update_local_best(
            err, err_old, acc_list, nmi_list, f1_list, k_iter + 1,
            local_best_acc, local_best_nmi, local_best_f1, local_best_iter,
            k_max, tol)

        if k_iter > 520 or err < tol:
            print('local_best_acc: {:.4f}'.format(local_best_acc),
                  'local_best_nmi: {:.4f}'.format(local_best_nmi),
                  'local_best_f1: {:.4f}'.format(local_best_f1))
        # 检查收敛
        if err < tol:
            print(f'Converged at iteration {k_iter + 1}')
            break

    print(f'\nLocal Best Results: ')
    print(f'  ACC: {local_best_acc:.4f}')
    print(f'  NMI: {local_best_nmi:.4f}')
    print(f'  F1:  {local_best_f1:.4f}')

    return Z_k, C_k, acc_list, nmi_list, f1_list


def solve_C(alpha_XX, I, beta, gamma, s, Z_k, lamb, VVT, C_prev=None):
    """
    Find the closed-form solution for C
    Now we have considered the term beta ||C - VV^T||_F^2 and the term using the sign information of C_prev.

    Parameter Explanation:
      alpha_XX : ndarray (n,n)    # alpha * X^T X
      I        : ndarray (n,n)    # Identity matrix
      beta     : float
      gamma    : float
      s        : float
      Z_k      : ndarray (n,n)
      lamb     : float            # Diagonal regularization term weight
      VVT      : ndarray (n,n) or None  # V @ V.T，
      C_prev   : ndarray or None  # The last time C，

    Return:
      C (ndarray n x n)
    """
    n = I.shape[0]

    #  G = α X^T X + (β + γ s^2) I
    G = alpha_XX + (beta + gamma * (s * s)) * I

    # The molecular term of W: N = α X^T X + β V V^T + γ s Z ⊙ sgn(C_prev)
    if C_prev is None:
        sign_term = Z_k  #  gamma * s * Z_k
    else:
        sign_term = Z_k * np.sign(C_prev)  # elementwise multiply (Z ⊙ sgn(C_prev))

    N = alpha_XX + beta * VVT + gamma * s * sign_term  # W

    # Inverse correlation: Prioritize using Cholesky to enhance stability
    try:
        L = np.linalg.cholesky(G)
        y = np.linalg.solve(L, N)
        W = np.linalg.solve(L.T, y)   # W = G^{-1} N
        y_eye = np.linalg.solve(L, I)
        V = np.linalg.solve(L.T, y_eye)  # V : G^{-1}
    except np.linalg.LinAlgError:
        W = np.linalg.solve(G, N)  # W = G^{-1} N
        V = np.linalg.inv(G)   # V : G^{-1}

    # Construct C column-wise, maintaining your original diagonal contraction form (using the diagonal elements of lamb and V)
    C = np.empty((n, n), dtype=float)
    eps = 1e-12
    for j in range(n):
        w_j = W[:, j]
        v_j = V[:, j]
        # The diagonal elements closed-form: c_jj = w_jj / (1 + lamb * v_jj)
        denom = 1.0 + lamb * v_j[j]
        if abs(denom) < eps:
            denom = eps if denom >= 0 else -eps
        c_jj = w_j[j] / denom
        c_j = w_j - lamb * c_jj * v_j
        C[:, j] = c_j
    return C



def calculate_s(A, C):
    """
    calculate s = ||A||_F / ||(|C| + |C.T|)/2||_F
    """
    numerator = np.linalg.norm(A, ord='fro')
    denominator_matrix = (np.abs(C) + np.abs(C.T)) / 2
    denominator = np.linalg.norm(denominator_matrix, ord='fro')

    if denominator == 0:
        raise ValueError("The denominator is zero, so s cannot be calculated.")
    s = numerator / denominator
    return s


def solve_Z_inner(A, U, S_p, beta, gamma, s, eta, n):
    P = np.dot(U, U.T)
    numerator = A + beta * P + gamma * s * S_p
    denominator = 1.0 + beta + gamma
    Y = numerator / denominator
    Y_pos = np.maximum(Y, 0)
    Z = shrinkz(Y_pos, eta, n)
    return Z

def shrinkz(Z, eta, n):
    eta_1 = int(eta / 2)
    mask_upper = np.triu(np.ones((n, n)), k=1).astype(bool)
    vals_upper = Z[mask_upper]
    abs_vals = np.abs(vals_upper)
    if len(abs_vals) > eta_1 and eta_1 > 0:
        threshold = np.partition(abs_vals, -eta_1)[-eta_1]
    else:
        threshold = 0.0
    # Construct the retention mask: retain the positions on the diagonal and in the upper triangle where the absolute value is greater than or equal to the threshold
    keep = np.zeros_like(Z, dtype=bool)
    np.fill_diagonal(keep, True)
    keep_upper = (mask_upper & (np.abs(Z) >= threshold))
    keep = keep | keep_upper | keep_upper.T
    Z_truncated = np.where(keep, Z, 0.0)
    Z_truncated = (Z_truncated + Z_truncated.T) / 2.0
    return Z_truncated