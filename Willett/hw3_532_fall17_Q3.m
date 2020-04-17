%% ECE 532 - Fall 2017 - HW3 - Q3
% by Roumen Guha

clear all
close all

load('hw3_532_fall17_face_emotion_data.mat');

%% (a) 
% Use the training data $X$ and $y$ to find a good set of weights.

w_hat = X\y

%% (b) 
% *How would you use these weights to classify a new face image as happy or mad?*
% Compute the vector $\hat{y}$. Classify values that are $> 0$ as happy, and
% values that are $< 0$ as mad. For values that are equal to 0, we flip a coin.
% But that seems unlikely with floating point numbers.

classify = @(x) sum([1*(x>0), -1*(x<0)], 2);                    % Classify elements in our label y_hat

y_hat = X*w_hat
classify(y_hat)

%% (c) 
% *Which features seem to be most important? Justify your answer.*
% Looking at the vector $\hat{w}$, it seems obvious that the first feature is
% the most important as it has the greatest weight. The second most important
% feature seems to be the fourth feature, as it is the most negative, so it
% could be considered instrumental in classifying an expression as mad.

%% (d) 
% *Can you design a classifier based on just 3 of the 9 features? Which 3 would you choose? How would you build a classifier?*
% I'd choose $w_1 = 0.9437$, $w_4 = -0.3922$ and $w_3 = 0.2664$. I'd zero out
% the other weights so that they'd have no contribution to our labels
% $\hat{y}$, and then simply do what we did for part (b) above, and if the value
% of $\hat{y_i}$ is $> 0$, we call it happy, and if it's $< 0$ we call it mad.
% If it's zero, we flip a coin.

%% (e) 
% *A common method for estimating the performance of a classifier is cross-validation (CV).
% CV works like this. Divide the dataset into 8 equal sized subsets (e.g., examples 1-16,
% 17-32, etc). Use 7 sets of the data to choose your weights, then use the weights to
% predict the labels of the remaining "hold-out" set. Compute the number of mistakes
% made on this hold-out set and divide that number by 16 (the size of the set) to estimate
% the error rate. Repeat this process 8 times (for the 8 different choices of the hold-out
% set) and average the error rates to obtain a final estimate.*

errors9 = zeros(8,1); % Prepare a vector of zeros

countErrors = @(x,y) sum(classify(x) ~= classify(y));           % Returns the number of errors in the set. 
errorPercentage = @(x,y) (countErrors(x,y) / length(x));        % Returns the percentage error of the set.

% for set k = 1
k = 1;
testY1 = y(1:end - 16*k);
testX1 = X(1:end - 16*k, :);
holdoutY1 = y(end - 16*k + 1 : end);
holdoutX1 = X(end - 16*k + 1 : end, :);

w_hat1 = testX1\testY1;
y_hat1 = holdoutX1*w_hat1;

errors9(k) = errorPercentage(y_hat1, holdoutY1);

% for sets k = 2:7
for k = 2:7
    testYk = [y(1:end - 16*k) ; y(end - 16*(k - 1) + 1 : end)];
    testXk = [X(1:end - 16*k, :) ; X(end - 16*(k - 1) + 1 : end, :)];
    holdoutYk = y(end - 16*k + 1 : end - 16*(k - 1));
    holdoutXk = X(end - 16*k + 1 : end - 16*(k - 1), :);
    
    w_hatk = testXk\testYk;
    y_hatk = holdoutXk*w_hatk;
    
    errors9(k) = errorPercentage(y_hatk, holdoutYk);
end

% for set k = 8
k = k + 1
testY8 = y(end - 16*k + 1 : end);
testX8 = X(end - 16*k + 1 : end, :);
holdoutY8 = y(1 : end - 16*(k - 1));
holdoutX8 = X(1 : end - 16*(k - 1), :);

w_hat8 = testX8\testY8;
y_hat8 = holdoutX8*w_hat8;

errors9(k) = errorPercentage(y_hat8, holdoutY8)

mean(errors9)

%% (f) 
% *What is the estimated error rate using all 9 features? What is it using the 3 features you chose in (d) above?*
% The mean error with 9 features is 0.0391.
% The mean error with 3 features is 0.0859.
% This seems right; more data helps the prediction to correctly place data it
% hasn't seen before. 

errors3 = zeros(8,1); % Prepare a vector of zeros

% for set k = 1
k = 1;
testY1 = y(1:end - 16*k);
testX1 = X(1:end - 16*k, :);
holdoutY1 = y(end - 16*k + 1 : end);
holdoutX1 = X(end - 16*k + 1 : end, :);

w_hat1 = testX1\testY1;
w_hat1 = [w_hat1(1) 0 w_hat1(3) w_hat1(4) 0 0 0 0 0]';
y_hat1 = holdoutX1*w_hat1;

errors3(k) = errorPercentage(y_hat1, holdoutY1);

% for sets k = 2:7
for k = 2:7
    testYk = [y(1:end - 16*k) ; y(end - 16*(k - 1) + 1 : end)];
    testXk = [X(1:end - 16*k, :) ; X(end - 16*(k - 1) + 1 : end, :)];
    holdoutYk = y(end - 16*k + 1 : end - 16*(k - 1));
    holdoutXk = X(end - 16*k + 1 : end - 16*(k - 1), :);
    
    w_hatk = testXk\testYk;
    w_hatk = [w_hatk(1) 0 w_hatk(3) w_hatk(4) 0 0 0 0 0]';
    y_hatk = holdoutXk*w_hatk;
    
    errors3(k) = errorPercentage(y_hatk, holdoutYk);
end

% for set k = 8
k = k + 1
testY8 = y(end - 16*k + 1 : end);
testX8 = X(end - 16*k + 1 : end, :);
holdoutY8 = y(1 : end - 16*(k - 1));
holdoutX8 = X(1 : end - 16*(k - 1), :);

w_hat8 = testX8\testY8;
w_hat8 = [w_hat8(1) 0 w_hat8(3) w_hat8(4) 0 0 0 0 0]';
y_hat8 = holdoutX8*w_hat8;

errors3(k) = errorPercentage(y_hat8, holdoutY8)

mean(errors3)
