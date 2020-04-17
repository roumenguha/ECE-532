function [x] = ista(H, y, lambda)
% L1-����ϡ��
% soft-���޺��� (ISTA)
% INPUT
% y - �۲����
% H - �ֵ����
% lambda - ���򻯱���
% alpha -L
% Nit - ��������
% OUTPUT
% x - ���ϡ�����
% J - Ŀ�꺯��
alpha = 0.5;
Nit = 10000;

J = zeros(1, Nit);
x = 0*H'*y; % ��ʼ�� x
T = lambda;
for k = 1:Nit%��������
    Hx = H*x;
    J(k) = sum(abs(Hx(:)-y(:)).^2) + lambda*sum(abs(x(:)));%����Ŀ�꺯��
    x = soft(x + (H'*(y - Hx))/alpha, T);%���������뵽ѵ��������
end