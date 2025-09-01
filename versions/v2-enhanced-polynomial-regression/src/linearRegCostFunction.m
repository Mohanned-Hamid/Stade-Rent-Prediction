%% LINEAR REGRESSION - RENT PREDICTION
% Author: Mohanned Hamid
% Date: 2025-09-01
% Description: Predicts apartment rents using size/rooms with improvement


function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


% Compute hypothesis
h = X * theta;

% Compute error
error = h - y;

% Compute cost
J = (1/(2*m)) * (error' * error) + (lambda/(2*m)) * (theta(2:end)' * theta(2:end));

% Compute gradient
grad = (1/m) * (X' * error);
grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);


grad = grad(:);

end
