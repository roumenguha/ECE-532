function [x] = ista(H, y, lambda)
% L1-限制稀疏
% soft-门限函数 (ISTA)
% INPUT
% y - 观测变量
% H - 字典矩阵
% lambda - 正则化变量
% alpha -L
% Nit - 迭代次数
% OUTPUT
% x - 输出稀疏编码
% J - 目标函数
alpha = 0.5;
Nit = 10000;

J = zeros(1, Nit);
x = 0*H'*y; % 初始化 x
T = lambda;
for k = 1:Nit%迭代次数
    Hx = H*x;
    J(k) = sum(abs(Hx(:)-y(:)).^2) + lambda*sum(abs(x(:)));%构建目标函数
    x = soft(x + (H'*(y - Hx))/alpha, T);%将输入输入到训练函数中
end