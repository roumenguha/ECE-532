function U = GS_orthogonalization(A)
% ECE 532 HW3 U2
%   U is the matrix of the Gram-Schmidt orthogonalized vectors spanning A's
%   vectors

[m, n] = size(A);

U = zeros(m, n);
R = zeros(n);

R(1, 1) = norm(A(:, 1));
U(:, 1) = A(:, 1)/R(1, 1);

for k = 2:n
    R(1:k-1, k) = U(:, 1:k-1)' * A(:, k);
    U(:, k) = A(:, k) - U(:, 1:k-1) * R(1:k-1, k);
    R(k, k) = norm(U(:, k));
    
    if R(k, k) == 0
        break;
    end
    
    U(:, k) = U(:, k) / R(k, k);
end
end