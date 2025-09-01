%% LINEAR REGRESSION - RENT PREDICTION
% Author: Mohanned Hamid
% Date: 2025-09-01
% Description: Predicts apartment rents using size/rooms with improvement

%% LINEAR REGRESSION - RENT PREDICTION WITH BIAS-VARIANCE ANALYSIS
% Author: Mohanned Hamid
% Date: 2025-08-14
% Description: Predicts apartment rents using size/rooms with bias-variance analysis

%% Clear and Close Figures
clear; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('Stade_data.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

%% ======================= Project Header =======================
fprintf('==================================================\n');
fprintf(' Rent Prediction in Stade, Germany - Bias-Variance Analysis\n');
fprintf('==================================================\n');
fprintf(' Features: Apartment Size (m²) + Number of Rooms\n');
fprintf(' Training Samples: %d\n', m);
fprintf(' Algorithm: Polynomial Regression with Regularization\n');
fprintf(' Developed by: Mohanned Hamid\n');
fprintf('==================================================\n\n');

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Split data into training, validation, and test sets
rng(1); % For reproducibility
indices = randperm(m);
train_ratio = 0.6;
val_ratio = 0.2;
test_ratio = 0.2;

train_size = round(train_ratio * m);
val_size = round(val_ratio * m);
test_size = m - train_size - val_size;

X_train = X(indices(1:train_size), :);
y_train = y(indices(1:train_size));

X_val = X(indices(train_size+1:train_size+val_size), :);
y_val = y(indices(train_size+1:train_size+val_size));

X_test = X(indices(train_size+val_size+1:end), :);
y_test = y(indices(train_size+val_size+1:end));

fprintf('Data split: Train=%d, Validation=%d, Test=%d\n', train_size, val_size, test_size);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Generate polynomial features and normalize
fprintf('Generating polynomial features and normalizing...\n');

% Generate polynomial features for all sets
p = 8; % Start with degree 8 as in ex5
X_poly = polyFeatures(X, p);
X_poly_val = polyFeatures(X_val, p);
X_poly_test = polyFeatures(X_test, p);

% Normalize features using training set statistics
[X_poly, mu, sigma] = featureNormalize(X_poly);
X_poly_val = (X_poly_val - mu) ./ sigma;
X_poly_test = (X_poly_test - mu) ./ sigma;

% Add intercept term
X_poly = [ones(m, 1), X_poly];
X_poly_val = [ones(val_size, 1), X_poly_val];
X_poly_test = [ones(test_size, 1), X_poly_test];

fprintf('Normalized training example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Train polynomial regression with different lambda values
fprintf('Training polynomial regression with different lambda values...\n');

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);

    % Train the model
    theta = trainLinearReg(X_poly, y, lambda);

    % Compute training error (without regularization)
    error_train(i) = linearRegCostFunction(X_poly, y, theta, 0);

    % Compute validation error (without regularization)
    error_val(i) = linearRegCostFunction(X_poly_val, y_val, theta, 0);

    fprintf('lambda = %f, Train Error = %f, Validation Error = %f\n', ...
            lambda, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Plot validation curve
fprintf('Plotting validation curve...\n');

figure;
semilogx(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
title('Validation Curve for Polynomial Regression');

% Find the best lambda
[~, best_idx] = min(error_val);
best_lambda = lambda_vec(best_idx);
fprintf('Best lambda found: %f\n', best_lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Train final model with best lambda
fprintf('Training final model with lambda = %f...\n', best_lambda);

theta = trainLinearReg(X_poly, y, best_lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Plot learning curves
fprintf('Plotting learning curves...\n');

[error_train_learn, error_val_learn] = learningCurve(X_poly, y, X_poly_val, y_val, best_lambda);

figure;
plot(1:m, error_train_learn, 1:m, error_val_learn);
title(sprintf('Learning Curve for Polynomial Regression (lambda = %f)', best_lambda));
xlabel('Number of training examples');
ylabel('Error');
legend('Train', 'Cross Validation');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Evaluate on test set
fprintf('Evaluating on test set...\n');

% Compute test error
test_error = linearRegCostFunction(X_poly_test, y_test, theta, 0);
fprintf('Test error: %f\n', test_error);

% Predict on test set
predictions = X_poly_test * theta;
mae = mean(abs(predictions - y_test));
fprintf('Mean Absolute Error on test set: €%.2f\n', mae);

% R-squared
SS_res = sum((y_test - predictions).^2);
SS_tot = sum((y_test - mean(y_test)).^2);
r2 = 1 - (SS_res / SS_tot);
fprintf('R-squared on test set: %.4f\n', r2);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Example predictions
fprintf('Making example predictions...\n');

% Example samples to predict
samples = [
    68, 2;    % 68 sq-m, 2 rooms
    85, 3;    % 85 sq-m, 3 rooms
    100, 4;   % 100 sq-m, 4 rooms
    120, 4;   % 120 sq-m, 4 rooms
    55, 1     % 55 sq-m, 1 room
];

fprintf('\nSample predictions:\n');
fprintf('Size (m²)\tRooms\tPredicted Rent\n');

for i = 1:size(samples, 1)
    sample = samples(i, :);
    sample_poly = polyFeatures(sample, p);
    sample_poly_norm = (sample_poly - mu) ./ sigma;
    sample_poly_norm = [1, sample_poly_norm];
    price = sample_poly_norm * theta;

    fprintf('%d\t\t%d\t€%.2f\n', sample(1), sample(2), price);
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Optional: Try different polynomial degrees
fprintf('Trying different polynomial degrees...\n');

p_values = [1, 2, 3, 4, 5, 6, 7, 8];
best_errors = zeros(length(p_values), 1);

for p_idx = 1:length(p_values)
    p = p_values(p_idx);

    % Generate polynomial features
    X_poly_p = polyFeatures(X, p);
    X_poly_val_p = polyFeatures(X_val, p);

    % Normalize features
    [X_poly_p, mu_p, sigma_p] = featureNormalize(X_poly_p);
    X_poly_val_p = (X_poly_val_p - mu_p) ./ sigma_p;

    % Add intercept term
    X_poly_p = [ones(m, 1), X_poly_p];
    X_poly_val_p = [ones(val_size, 1), X_poly_val_p];

    % Train model with best lambda from previous step
    theta_p = trainLinearReg(X_poly_p, y, best_lambda);

    % Compute validation error
    best_errors(p_idx) = linearRegCostFunction(X_poly_val_p, y_val, theta_p, 0);

    fprintf('p = %d, Validation Error = %f\n', p, best_errors(p_idx));
end

% Find best polynomial degree
[~, best_p_idx] = min(best_errors);
best_p = p_values(best_p_idx);
fprintf('Best polynomial degree: %d\n', best_p);

figure;
plot(p_values, best_errors);
xlabel('Polynomial Degree');
ylabel('Validation Error');
title('Validation Error vs. Polynomial Degree');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Final model with best parameters
fprintf('Training final model with best parameters (p=%d, lambda=%f)...\n', best_p, best_lambda);

% Generate polynomial features with best degree
X_poly_best = polyFeatures(X, best_p);
X_poly_test_best = polyFeatures(X_test, best_p);

% Normalize features
[X_poly_best, mu_best, sigma_best] = featureNormalize(X_poly_best);
X_poly_test_best = (X_poly_test_best - mu_best) ./ sigma_best;

% Add intercept term
X_poly_best = [ones(m, 1), X_poly_best];
X_poly_test_best = [ones(test_size, 1), X_poly_test_best];

% Train final model
theta_final = trainLinearReg(X_poly_best, y, best_lambda);

% Evaluate on test set
test_error_final = linearRegCostFunction(X_poly_test_best, y_test, theta_final, 0);
fprintf('Final test error: %f\n', test_error_final);

predictions_final = X_poly_test_best * theta_final;
mae_final = mean(abs(predictions_final - y_test));
fprintf('Final MAE: €%.2f\n', mae_final);

SS_res_final = sum((y_test - predictions_final).^2);
SS_tot_final = sum((y_test - mean(y_test)).^2);
r2_final = 1 - (SS_res_final / SS_tot_final);
fprintf('Final R-squared: %.4f\n', r2_final);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Visualization: Regression surface
fprintf('Plotting regression surface...\n');

% Use original data for plotting
figure;
scatter3(X(:,1), X(:,2), y, 'filled');
hold on;

% Create grid for surface plot
x1_range = linspace(min(X(:,1)), max(X(:,1)), 50);
x2_range = linspace(min(X(:,2)), max(X(:,2)), 50);
[X1_grid, X2_grid] = meshgrid(x1_range, x2_range);

% Generate polynomial features for grid points
X_grid_poly = polyFeatures([X1_grid(:), X2_grid(:)], best_p);
X_grid_poly_norm = (X_grid_poly - mu_best) ./ sigma_best;
X_grid_poly_norm = [ones(size(X_grid_poly_norm,1), 1), X_grid_poly_norm];

% Predict values for grid
y_grid = X_grid_poly_norm * theta_final;
y_grid = reshape(y_grid, size(X1_grid));

% Plot surface
mesh(X1_grid, X2_grid, y_grid);
title(sprintf('Regression Surface (p=%d, λ=%.3f)', best_p, best_lambda));
xlabel('Size (m²)');
ylabel('Rooms');
zlabel('Rent (€)');
legend('Data Points', 'Regression Surface');
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;

%% End of script
fprintf('Rent prediction project with bias-variance analysis completed.\n');
fprintf('Best parameters: p=%d, lambda=%.3f\n', best_p, best_lambda);
fprintf('Final test error: %.2f\n', test_error_final);
