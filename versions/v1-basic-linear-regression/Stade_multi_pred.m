%% LINEAR REGRESSION - RENT PREDICTION
% Author: Mohanned Hamid
% Date: 2025-08-14
% Description: Predicts apartment rents using size/rooms


%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('Stade_data.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

%% ======================= Project Header =======================
fprintf('==================================================\n');
fprintf(' Rent Prediction in Stade, Germany - Linear Regression\n');
fprintf('==================================================\n');
fprintf(' Features: Apartment Size (m²) + Number of Rooms\n');
fprintf(' Training Samples: %d\n', m);
fprintf(' Algorithm: Gradient Descent & Normal Equation\n');
fprintf(' Developed by: Mohanned Hamid\n');
fprintf('==================================================\n\n');


% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1), X];



%% ================ Part 2: Gradient Descent ================


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.09;
num_iters = 400;

% Init Theta and Run Gradient Descent
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 68 sq-m, 2 br house
% ======================  THE CODE  ===========================
% The first column of X is all-ones. Thus, it does
% not need to be normalized.
x = [68 2];
x = (x-mu)./sigma;
x = [1 x];
price = x * theta;
% ============================================================
fprintf(['Predicted the price of a 68 sq-m, 2 br house ' ...
         '(using gradient descent):\n €%.2f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ======================  THE CODE  ===========================


%% Load Data
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 68 sq-m, 2 br house
% ======================  THE CODE  ===========================
x = [1 68 2];
price = x * theta;


% ============================================================

fprintf(['Predicted the price of a 68 sq-m, 2 br house ' ...
         '(using normal equations):\n €%.2f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 4: Visualization ================

fprintf('Plot regression_surface...\n');

figure;
raw_data = load('Stade_data.txt');
scatter3(raw_data(:,1), raw_data(:,2), raw_data(:,3), 'filled');
hold on;
x1fit = linspace(min(X(:,2)), max(X(:,2)), 100);
x2fit = linspace(min(X(:,3)), max(X(:,3)), 100);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = theta(1) + theta(2)*X1FIT + theta(3)*X2FIT;
mesh(X1FIT,X2FIT,YFIT);
title('Regression Surface');
xlabel('Size (m²)');
ylabel('Rooms');
zlabel('Rent (€)');

%% ================ Part 5: Model Evaluation ================
fprintf('Evaluating model performance...\n');

% Mean Absolute Error (MAE)
predictions = X * theta;
mae = mean(abs(predictions - y));
fprintf('Mean Absolute Error: €%.2f\n', mae);

%  R-squared
SS_res = sum((y - predictions).^2);
SS_tot = sum((y - mean(y)).^2);
r2 = 1 - (SS_res / SS_tot);
fprintf('R-squared: %.4f\n', r2);

