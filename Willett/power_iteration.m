function [vector, value] = power_iteration(A, num_iterations, start)
% Power method for computing eigenvalues

if nargin < 3
    start = 0;
end

if nargin < 2
    num_iterations = 1000000000000000;
end

if nargin < 1
    error('Matrix A is a required input!')
    return;
elseif start == 0
    [n,m] = size(A);
    if n ~= m
        disp('Matrix A is not square!');
        return;
    end
    
    start = rand(n,1);
end

x = start;

while num_iterations > 0
    x = A*x;
    x = x/norm(x);
    num_iterations = num_iterations - 1;
end
vector = x;
value = norm(x);

end
