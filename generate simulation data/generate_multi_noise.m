clc; clear;
rng(66); 

% 定义参数
mu1 = [1; 1; 0; 0; 0];
mu2 = [0; 0; 1; 0; 0];
mu3 = [0; 0; 0; 1; 1];


% 组合成 d×3 矩阵
para.centroids = [mu1, mu2, mu3];               % 边扰动比例
para.ns = [50, 80, 100];         % 各类样本数
%% -----------------------------
%  参数设置
%% -----------------------------
eps_list   = 0:0.1:1.0;
delta_list = 0:0.1:1.0;
para.epsilon = 0;
para.delta   = 0;

%% -----------------------------
%  生成基础图
%% -----------------------------
[X0, labels, A0, ~] = gen_attributed_graph_from_centroid(para);

%% -----------------------------
%  生成带噪声 X
%% -----------------------------
for i = 1:length(eps_list)
    para.epsilon = eps_list(i);
    para.delta   = 0; % 不影响 X

    [X, ~, ~, ~] = gen_attributed_graph_from_centroid(para);

    % 构建变量名，例如 X0, X0_1, X0_2, ..., X1
    if eps_list(i) == 0
        varname = 'X0';
    elseif eps_list(i) == 1
        varname = 'X1';
    else
        % 把 0.1 -> _1, 0.2 -> _2, ...
        idx = round(eps_list(i)*10);
        varname = ['X0_' num2str(idx)];
    end
    eval([varname ' = X;']);
end

%% -----------------------------
%  生成带噪声 A
%% -----------------------------
for j = 1:length(delta_list)
    para.epsilon = 0; % 不影响 A
    para.delta   = delta_list(j);

    [~, ~, ~, A] = gen_attributed_graph_from_centroid(para);

    % 构建变量名，例如 A0, A0_1, A0_2, ..., A1
    if delta_list(j) == 0
        varname = 'A0';
    elseif delta_list(j) == 1
        varname = 'A1';
    else
        idx = round(delta_list(j)*10);
        varname = ['A0_' num2str(idx)];
    end
    eval([varname ' = A;']);
end

%% -----------------------------
%  保存数据
%% -----------------------------
save('graph_noise_epsilon_square2.mat', ...
     'X0', 'X0_1','X0_2','X0_3','X0_4','X0_5','X0_6','X0_7','X0_8','X0_9','X1', ...
     'A0', 'A0_1','A0_2','A0_3','A0_4','A0_5','A0_6','A0_7','A0_8','A0_9','A1', ...
     'labels', 'A0', 'para', '-v7');