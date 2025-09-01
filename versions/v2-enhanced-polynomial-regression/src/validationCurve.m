%% LINEAR REGRESSION - RENT PREDICTION
% Author: Mohanned Hamid
% Date: 2025-09-01
% Description: Predicts apartment rents using size/rooms with improvement

function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)

% Selected values of lambda,
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    % Train the model with the current lambda
    theta = trainLinearReg(X, y, lambda);
    % Compute training error (without regularization)
    error_train(i) = linearRegCostFunction(X, y, theta, 0);
    % Compute validation error (without regularization)
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end


end
