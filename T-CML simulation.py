from numpy.linalg import inv, solve
import numpy as np  # Import the NumPy library for numerical computing
from scipy import linalg
from scipy.sparse import eye, issparse  # Import the eye function from the SciPy library for creating sparse identity matrices
from scipy.sparse.linalg import eigs  # Import the eigs function from the SciPy library for computing eigenvalues and eigenvectors of sparse matrices
from utiles import clustering_metrics, update_local_best, spectral_clusteringA_mean, TCML_simulation_params
from sklearn.preprocessing import StandardScaler


def TCML(A, X, gnd, eta, n_class, alpha, beta, lamb, gamma, tau, max_iter=10, initialize=True):
    n = A.shape[0]  # Get the number of rows in the input matrix A
    norm = np.linalg.norm(A, ord=None)
    scale = np.sqrt(n_class) / norm #Compute a scaling factor based on the number of classes and the Frobenius norm of A
    As = (A + A.T) * (scale / 2)  # Scale and symmetrize the input matrix A
    I = np.eye(n)
    if issparse(X):
        X = X.todense()
    Xnorm = np.linalg.norm(X, ord=None)
    scaleX = np.sqrt(n_class) / Xnorm
    X = X * scaleX
    alpha_XX = alpha * np.dot(X, X.T)

    if initialize == True:
        Z = As
        Y = np.zeros((n, n))
    elif initialize == "L1":
        G, _ = compute_graph_matrix(X.T, alpha=5, max_iters=100)
        Z = G + np.eye(n)
        Y = Z

    acc = []
    nmi = []
    F1 = []
    err_list=[]
    local_best_acc = 0.0
    local_best_nmi = 0.0
    local_best_f1 = 0.0
    local_best_iter = 0
    err = 1  # Initialize the error to be greater than the convergence tolerance
    iter = 0  # Initialize the iteration counter

    ac, _, nm, _, f1, _= spectral_clusteringA_mean(As, gnd, n_iter=50, k=3)
    print('iter: {}'.format(iter), 'acc: {:.4f}'.format(ac), 'nmi: {:.4f}'.format(nm))

    while err > tau and iter < max_iter:  # While the error is greater than the convergence tolerance and the maximum number of iterations has not been reached,
        iter += 1  # increment the iteration counter
        Z_old = Z  # store the current value of Z
        err_old = err

        # Update U
        try:
            U = eigs(Y, k=n_class, which='LM')[1]
            U = np.real(U)
        except Exception as e:
            U = U_old
        UU = np.real(np.dot(U, U.T))

        # Update C
        AAA = (beta + 1 + gamma) * np.eye(n) + alpha_XX
        BBB = As + beta * UU + alpha_XX + gamma * Z_old - lamb * I
        C = linalg.solve(AAA, BBB)
        C = C.astype(float)
        D = C * (C > 0)

        # Update Z to T_eta(Z,D) by truncating C:
        Z = shrink(C, D, eta, n)

        err = np.linalg.norm(Z - Z_old, ord=None)
        err = err / np.linalg.norm(Z, ord=None)
        err_list.append(err)
        print('err: {:.4f}'.format(err),  'iter: {}'.format(iter))

        Z_sym = (np.abs(Z) + np.abs(Z.T)) / 2
        U_old = U.copy()
        try:
            ac, _, nm, _, f1, _ = spectral_clusteringA_mean(Z_sym, gnd, n_iter=2, k=3)
        except Exception as e:
            print(f"[Warning] Evaluation failed at iteration {iter}: {e}")
            ac, nm, f1 = -1, -1, -1
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
    return Z, U, acc, nmi, F1  # Return the final values of Z, U, n_inners, and (optionally) Zs


def shrink(Z, D, eta, n):  # Define a helper function for truncating a matrix
    eta_1 = int(eta / 2)  # Compute half of eta rounded down to an integer
    dd = np.triu(D, 1).flatten()  # Extract the upper triangular part of D above its main diagonal and flatten it into a one-dimensional array
    dd.sort()  # Sort dd in ascending order
    dd = dd[::-1]  # Reverse dd to obtain it in descending order
    return Z * ((np.eye(n) + (D >= dd[eta_1])) != 0)


def has_nonzero_diagonal(A):
    """
    Check whether there are non-zero elements on the diagonal of matrix A
    :param A: Input matrix
    :return: If there are non-zero elements on the diagonal, return True; otherwise, return False
    """
    diagonal_elements = np.diag(A)
    return np.any(diagonal_elements != 0)

def compute_graph_matrix(X, alpha=1e-3, max_iters=1000, tol=1e-6):
    """
    Unified self-representation graph matrix computation function
    Parameters
    ----------
    X : ndarray of shape (d, n)
    Input data matrix, d features, n samples (each column is a sample)
    For example: (5, 230) method : str, {"l2_diag", "l2_trace", "l2_standard", "l1_lasso", "l1_proximal"}
    Self-expression model selection:
    "l1_proximal" -> min ||X - XC||_F^2 + α ||C||_1 (using proximal gradient method) alpha : float, default=1e-3
    Regularization parameter, controlling the strength of regularization max_iters : int, default=1000
    Maximum number of iterations (only for the l1_proximal method) tol : float, default=1e-6
    Convergence tolerance (used only for the l1_proximal method)
    Returns
    -------
    G : ndarray of shape (n, n)
    Symmetric adjacency matrix G = 0.5 * (|C| + |C|^T) C : ndarray of shape (n, n)
    Self-representation coefficient matrix
    """

    X = np.array(X, dtype=float)
    if X.shape[0] > X.shape[1]:
        X = X.T
    d, n = X.shape
    print(f"Data shape: ({d}, {n}) - {d} features, {n} samples")

    X_normalized = StandardScaler().fit_transform(X.T).T  # (d, n)
    XtX = X_normalized.T @ X_normalized  # (n, n)

    C = np.zeros((n, n))
    I = np.eye(n)

    # Lipschitz constant
    L = np.linalg.norm(XtX, 2)

    for iteration in range(max_iters):
        # Gradient: ∇f(C) = 2(X^T X C - X^T X)
        grad = 2 * (XtX @ C - XtX)

        # Gradient Descent
        C_grad = C - (1 / L) * grad

        # Proximal operator (soft thresholding)
        C_new = np.sign(C_grad) * np.maximum(np.abs(C_grad) - alpha / L, 0)

        # Require the diagonal elements to be zero.
        np.fill_diagonal(C_new, 0)
        C = C_new
    G = 0.5 * (np.abs(C) + np.abs(C.T))
    return G, C




if __name__ == '__main__':
    import scipy.io as sio
    mat_path = 'data/graph_noise_epsilon_square2.mat'
    mat_data = sio.loadmat(mat_path)
    A_name = 'A0_9'
    X_name = 'X0_1'
    A = mat_data[A_name]
    X = mat_data[X_name].T
    gnd = mat_data['labels'].flatten()
    k = 3
    A_val = A_name[1:].replace('_', '.')
    X_val = X_name[1:].replace('_', '.')
    key = f"A{A_val}+X{X_val}"
    acml_params = TCML_simulation_params[key]


    Z, U, acc, nmi,F1 = TCML(A, X, gnd, **acml_params)
    print('acc: {:.4f}'.format(acc[-1]),
         'nmi: {:.4f}'.format(nmi[-1]),
          'F1: {:.4f}'.format(F1[-1]))

