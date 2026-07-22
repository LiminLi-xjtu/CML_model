function [X, labels, A0, A] = gen_attributed_graph_from_centroid(para)
%GEN_ATTRIBUTED_GRAPH_FROM_CENTROID
%
%  Generate an attributed graph with node features from
%  approximate 1D subspaces and kNN graph structure.
%
%  Input:
%    centroid : d×3 matrix, subspace directions
%    epsilon  : noise level in node features
%    k        : number of nearest neighbors (kNN)
%    delta    : edge perturbation ratio (0 ≤ delta ≤ 1)
%
%  Output:
%    X      : d×230 node feature matrix
%    labels : 1×230 node class labels
%    A0     : original kNN adjacency matrix
%    A      : perturbed adjacency matrix

    %% -----------------------------
    %  1. Generate node features
    %% -----------------------------
    
    centroid = para.centroids;
    epsilon = para.epsilon;
    delta = para.delta;
    ns = para.ns;
    n = sum(ns);
    
    % [X, labels] = gen_1d_subspace_from_centroid(centroid, epsilon,ns);
    [X, X_clean, labels] = gen_1d_subspace_from_centroid(centroid, epsilon, ns);
    N = size(X, 2);

    % Z_clean = pca(X_clean);   % MATLAB 的 pca 默认返回 score
    % figure;
    % gscatter(Z_clean(:,1), Z_clean(:,2), labels(:), 'rgb', 'o', 6);
    % axis equal; grid on;
    % title('Clean node features (before noise)');
    % xlabel('PC 1'); ylabel('PC 2');

    Z_noisy = pca(X);
    figure;
    
    gscatter(Z_noisy(:,1), Z_noisy(:,2), labels', 'rgb', 'o', 7);
    axis equal; grid on;
    title(['Noisy node features (\epsilon = ', num2str(epsilon), ')'], 'FontSize', 16);
    xlabel('PCA 1', 'FontSize', 14);
    ylabel('PCA 2', 'FontSize', 14);
    set(gca, 'FontSize', 13);
    lgd = legend({'Class 1', 'Class 2', 'Class 3'});
    lgd.FontSize = 13;


    % % 原始 clean 特征可视化
    % figure;
    % gscatter(X_clean(1,:), X_clean(2,:), labels(:), 'rgb', 'o', 6);
    % axis equal; grid on;
    % title('Clean node features (before noise)');
    % xlabel('Feature 1'); ylabel('Feature 2');
    % 
    % % 加噪声后的特征可视化
    % figure;
    % gscatter(X(1,:), X(2,:), labels(:), 'rgb', 'o', 6);
    % axis equal; grid on;
    % title(['Noisy node features (after noise, \epsilon = ', num2str(epsilon), ')']);
    % xlabel('Feature 1'); ylabel('Feature 2');

    %% -----------------------------
    %  2. Build kNN graph (A0)
    %% -----------------------------
    % pairwise Euclidean distances
    % D = pdist2(X', X');
    % % D = pdist2(X_clean', X_clean');
    % 
    % % initialize adjacency
    % A0 = eye(N, N);
    % k = 25;
    % for i = 1:N
    %     % sort distances (exclude self-loop)
    %     [~, idx] = sort(D(i, :), 'ascend');
    %     neighbors = idx(2:k+1);
    %     A0(i, neighbors) = 1;
    % end
    % 
    % A_mask = blkdiag( ...
    %     ones(ns(1)), ...
    %     ones(ns(2)), ...
    %     ones(ns(3)) ...
    % );
    % 
    % % symmetrize (undirected graph)
    % A_knn = double((A0 + A0') > 0);
    % A0 = A_knn .* A_mask;


    %% -----------------------------
    % 2. Build clean block-sparse graph (A0)
    %% -----------------------------
    p = 0.05;
    A0 = zeros(N, N);
    
    start_idx = 1;
    for k = 1:length(ns)
        nk = ns(k);
        idx = start_idx : start_idx + nk - 1;
    
        % 只在上三角采样（每一条无向边只采样一次）
        Ak = zeros(nk, nk);
        mask = triu(true(nk), 1);
        Ak(mask) = rand(nnz(mask), 1) < p;
    
        % 对称化
        Ak = Ak + Ak';
    
        % 填回块对角
        A0(idx, idx) = Ak;
    
        start_idx = start_idx + nk;
    end


    %% -----------------------------
    %  3. Random edge perturbation
    %% -----------------------------
    
    
    % delta_half = floor(delta * nnz(A0)/2);
    delta_half = floor(delta * delta * nnz(A0)/2);
    A_c = 1 - blkdiag(ones(ns(1)), ones(ns(2)), ones(ns(3)));
    a = triu(A_c,1); a = a(:);
    idx_a = find(a==1);
    p = randperm(nnz(a),delta_half);
    b = zeros(size(a)); b(idx_a(p)) = 1;
    B_c = reshape(b,n,n); A_c = B_c + B_c';

    A = A0 + A_c;



% % % % %     A = A0;
% % % % % 
% % % % %     % upper triangular indices (exclude diagonal)
% % % % %     mask = triu(ones(N), 1);
% % % % %     edge_idx = find(mask);
% % % % % 
% % % % %     num_edges = numel(edge_idx);
% % % % %     num_flip = round(delta * num_edges);
% % % % % 
% % % % %     % randomly select edges to flip
% % % % %     perm = randperm(num_edges, num_flip);
% % % % %     flip_idx = edge_idx(perm);
% % % % % 
% % % % %     % flip edges
% % % % %     for t = 1:num_flip
% % % % %         [i, j] = ind2sub([N, N], flip_idx(t));
% % % % %         A(i, j) = 1 - A(i, j);
% % % % %         A(j, i) = A(i, j);
% % % % %     end

    % remove self-loops
%     A(1:N+1:end) = 0;
end



function [X, X_clean, labels] = gen_1d_subspace_from_centroid(centroid, epsilon, ns)
% function [X, labels] = gen_1d_subspace_from_centroid(centroid, epsilon,ns)
%GEN_1D_SUBSPACE_FROM_CENTROID
%  Generate samples approximately lying in three 1D subspaces
%  defined by centroid vectors
%
%  Input:
%    centroid : d×3 matrix, each column defines a 1D subspace direction
%    epsilon  : noise level
%
%  Output:
%    X      : d×230 data matrix
%    labels : 1×230 subspace labels (1,2,3)

    % sample numbers
    

    % ambient dimension
    d = size(centroid, 1);

    % normalize centroids to get directions
    B = zeros(d, 3);
    for k = 1:3
        B(:,k) = centroid(:,k) / norm(centroid(:,k));
    end

    X = [];
    X_clean = [];
    labels = [];

    for k = 1:3
        nk = ns(k);
        bk = B(:,k);

        % coefficients along the subspace
        alpha = randn(1, nk);
        % alpha = 2*rand(1, nk);

        % clean 1D subspace data
        Xk = bk * alpha;

        % orthogonal complement
        Uk = null(bk');

        % perturbation (orthogonal noise)
        % sigma = std(Xk, 0, 2);      % d × 1，每个特征维度的 std
        % Noise = randn(size(Xk)) .* sigma;
        % Ek = epsilon * Noise;

        % Ek_raw = Uk * randn(size(Uk,2), nk); 
        % Ek = epsilon * norm(Xk,'fro') / norm(Ek_raw,'fro') * Ek_raw;

        Ek = epsilon * epsilon * Uk * randn(size(Uk,2), nk);
        % Ek = epsilon * Uk * randn(size(Uk,2), nk);

        % final samples
        X_clean = [X_clean, Xk];      % 无噪声
        X = [X, Xk + Ek];
        labels = [labels, k * ones(1, nk)];
    end
end


