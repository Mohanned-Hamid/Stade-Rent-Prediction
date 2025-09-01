%% LINEAR REGRESSION - RENT PREDICTION
% Author: Mohanned Hamid
% Date: 2025-09-01
% Description: Predicts apartment rents using size/rooms with improvement

function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for i = 1:m
    % Train on the first i examples
    X_train = X(1:i, :);
    y_train = y(1:i);
    theta = trainLinearReg(X_train, y_train, lambda);

    % Compute training error (without regularization)
    error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0);

    % Compute validation error (without regularization)
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end



end
