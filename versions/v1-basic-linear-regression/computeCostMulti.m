%% LINEAR REGRESSION - RENT PREDICTION
% Author: Mohanned Hamid
% Date: 2025-08-14
% Description: Predicts apartment rents using size/rooms

function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% theta: Parameters vector [n x 1]
% J_history: Cost function history [iterations x 1]

% Initialize some useful values
m = length(y); % number of training examples

% ============================== THE CODE  ==============================

prediction = X * theta;
squareErorr = (prediction -y).^2;

J = (1/(2*m))* sum(squareErorr) ;

% =========================================================================

end
