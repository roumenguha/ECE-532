%% ECE 532 - Fall 2017 - HW7

clear
close all

%% (1) Training Neural Networks.

p = 2;
n = 1e4;

% generate training data
X = rand(n, p) - 5;
Y1 = sum(X .^2, 2) > 1;
Y2 = 5*X(:,1).^3 > X(:,2);
Y = [Y1 Y2];

figure(1); clf;
subplot(121);
scatter(X(:,1), X(:,2), 20, Y1, 'filled');
title('training data, label 1');
axis image; colorbar; colormap jet; set(gca, 'fontsize', 18)
subplot(122);
scatter(X(:,1), X(:,2), 20, Y2, 'filled');
title('training data, label 2');
axis image; colorbar; colormap jet; set(gca, 'fontsize', 18)

Xb = [ones(n, 1) X];
q = size(Y, 2);
M = 2;
V = randn(M + 1, q);
W = randn(p + 1, M);

alpha = .1;

for epoch = 1:100
    ind = randperm(n);
    for i = ind
        % forward prop
        H = logsig([1 Xb(i ,:)*W]); % 1 x M+1
        
        Yhat = logsig(H*V); % 1 x q
        
        % backprop
        delta = (Yhat - Y(i,:)) .* Yhat .* (1 - Yhat); % 1 x q
        Vnew = V - alpha * H' * delta ;
        gamma = (delta * V(2:end,:)') .* H(2:end) .* (1 - H(2:end)); % 1 x M
        Wnew = W - alpha * Xb(i,:)' * gamma;
        V = Vnew;
        W = Wnew;
    end
    epoch
end

% final predicted labels
H = logsig([ones(n,1) Xb*W]); % n x M+1
Yhat = logsig(H*V); % n x q

figure(2); clf;
subplot(121); scatter(X(:,1), X(:,2), 20, Yhat(:,1), 'filled');
title('learned labels, label 1');
axis image; colorbar; colormap jet; set(gca, 'fontsize', 18)
subplot(122); scatter(X(:,1), X(:,2), 20, Yhat(:,2), 'filled');
title('learned labels, label 2');
axis image; colorbar; colormap jet; set(gca, 'fontsize', 18)

figure(3); clf;
subplot(121); scatter(X(:,1), X(:,2), 20, 1*(Yhat(:,1) > 5), 'filled');
title('thresholded learned labels, label 1');
axis image; colorbar; colormap jet; set(gca, 'fontsize', 18)
subplot(122); scatter(X(:,1), X(:,2), 20, 1*(Yhat(:,2) > 5), 'filled');
title ('thresholded learned labels, label 2');
axis image; colorbar; colormap jet; set(gca, 'fontsize', 18)

%% (a) Run the code. How does it perform?
% Are the learned labels close to the original lables?

% It's tough to say how the code performs; both saturate the space they're in
% because both are 10,000 data points. But it looks like it did really well, I
% can't tell the difference between the original data and the thresholded
% learned labels.

%% (b) Why do we use Xb instead of X? What if we use X instead?

% We use Xb to create an offset in the thesholded learned labels for feature 1.
% If we use X instead, the range of intensities for feature 1 is the same as
% that of feature 2. But if we use Xb, the intensities are in the correct range,
% as they are for the original data.

%% (c) Explain the use of the "2"s in the expression for gamma.

% We exclude the row of ones because they aren't used to calculate the weights,
% they simply serve to correct the range of outputs.

%% (d) Try increasing the number of epochs to 100.
% What effect does this have?



%% (e) Try increasing the number of hidden nodes to 3; what happens?
% What happens if you use 4 hidden nodes? Can you explain why four hidden
% nodes performs so much differently from two?


