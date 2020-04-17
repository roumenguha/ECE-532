function w = hw6_532_fall17_ista(X, y, lambda)
% ISTA Iterative Soft-Thresholding via Proximal Descent Algorithm
% The objective function: f(x) = 1/2 ||y - X*Beta||^2 + lambda*|x|
%
% Takes:
%       X: covariate matrix,
%       y: response vector,
%       lambda: penalty parameter, lambda > 0,
% Returns:
%       w: a (p x num_iters) matrix of values of weight vector Beta through
%       the iterations

MAX_ITER = 10000;
ABSTOL   = 1e-4;

[n, p] = size(X);
w = zeros(p, MAX_ITER);
w(:, 1) = rand(p, 1);

tau = 0.000000001/norm(X)^2; % 0 < tau < 2/norm(X)^2

for k = 1:MAX_ITER
    disp(k)
    
    z = w(:, k) - tau*X'*(X*w(:, k) - y);
    w(:, k+1) = wthresh(z, 's', lambda*tau/2);
    %norm(z - w(:, k)).^2 + lambda*tau*norm(w(:, k), 1);
    
    % Check for convergence
    if sum(abs(w(:, k+1) - w(:, k))) < ABSTOL
        break;
    end
    
end
w = w(:, k+1);
end