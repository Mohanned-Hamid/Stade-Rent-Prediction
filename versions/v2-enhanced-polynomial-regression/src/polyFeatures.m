%% LINEAR REGRESSION - RENT PREDICTION
% Author: Mohanned Hamid
% Date: 2025-09-01
% Description: Predicts apartment rents using size/rooms with improvement

function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (m x n matrix) into polynomial features of degree p
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x n) and
%   maps each example into its polynomial features up to degree p.

[m, n] = size(X);

% Initialize with original features
X_poly = X;

% Add polynomial features for each degree from 2 to p
for degree = 2:p
    for i = 1:n
        % Add each feature raised to the current degree
        X_poly = [X_poly, X(:, i).^degree];
    end

    % Add interaction terms between features
    if n > 1
        for i = 1:n-1
            for j = i+1:n
                % Add interaction terms (feature_i * feature_j)
                X_poly = [X_poly, X(:, i) .* X(:, j)];

                % Add higher-order interaction terms
                for d = 2:degree-1
                    X_poly = [X_poly, (X(:, i).^d) .* X(:, j)];
                    X_poly = [X_poly, X(:, i) .* (X(:, j).^d)];
                end
            end
        end
    end
end
end
