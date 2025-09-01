# 📊 Model Performance Results

## Comparative Analysis: Version 1 vs Version 2

### Performance Metrics Comparison

| Metric | Version 1 (Basic) | Version 2 (Enhanced) | Improvement |
|--------|-------------------|----------------------|-------------|
| **Mean Absolute Error** | €137.62 | €131.34 | **4.6% reduction** |
| **R-squared Score** | 0.6523 | 0.8207 | **25.8% improvement** |
| **Best Parameters** | N/A | p=8, λ=0.030 | Automated selection |

## Detailed Version 1 Results (Basic Linear Regression)

### Model Parameters
Gradient Descent Theta: [828.69, 267.99, -55.03]
Normal Equation Theta: [170.57, 11.25, -68.76]


### Prediction Examples
- 68 m², 2 rooms: **€797.97-797.99**
- Model Consistency: ✓ (Both methods give similar results)

### Performance
- **Mean Absolute Error**: €137.62
- **R-squared**: 0.6523
- **Interpretation**: Basic model explains 65.23% of rent variance

## Detailed Version 2 Results (Enhanced Polynomial Regression)

### Optimal Parameters
- **Polynomial Degree**: 8
- **Regularization Parameter (λ)**: 0.030
- **Final Cost**: 11,223.34 (after 200 iterations)

### Prediction Examples
| Size (m²) | Rooms | Predicted Rent |
|-----------|-------|----------------|
| 68 | 2 | €811.13 |
| 85 | 3 | €924.00 |
| 100 | 4 | €968.94 |
| 120 | 4 | €1069.48 |
| 55 | 1 | €738.13 |

### Performance
- **Test Error**: 13,520.55
- **Mean Absolute Error**: €131.34
- **R-squared**: 0.8207
- **Interpretation**: Enhanced model explains 82.07% of rent variance

## Technical Analysis

### Convergence Characteristics
- Cost function converged effectively after 200 iterations
- Regularization (λ=0.03) prevented overfitting while maintaining model flexibility

### Bias-Variance Tradeoff
- Version 1: Higher bias (underfitting visible in linear assumptions)
- Version 2: Better balance achieved through polynomial features and regularization

### Validation Results
- Optimal polynomial degree: 8 (determined through systematic testing)
- Best regularization parameter: λ=0.03 (optimized for generalization)

## Visualization Assets

Include these charts in your documentation:

1. **Cost Convergence Chart** (`cost_convergence.png`)
   - Shows model training progress
   - Demonstrates effective optimization

2. **Regression Surface 3D Plot** (`regression_surface.png`)
   - Visualizes complex relationships captured by polynomial model
   - Shows interaction between size and rooms

3. **Learning Curves** (`learning_curve.png`)
   - Demonstrates model learning progress
   - Shows training vs validation performance

4. **Validation Error Analysis** (`validation_curve.png`)
   - Shows parameter selection process
   - Illustrates bias-variance tradeoff

## Conclusion

The enhanced polynomial regression model (Version 2) demonstrates significant improvements over the basic linear regression approach (Version 1):

1. **Better Accuracy**: 4.6% reduction in prediction error
2. **Improved Explanatory Power**: 25.8% increase in R-squared value
3. **Automated Optimization**: Systematic parameter selection
4. **Enhanced Capability**: Captures non-linear relationships

This progression from basic to advanced techniques showcases effective application of machine learning principles to real-world problems.

---

*Last Updated: $(date)*  
*Model Trained on: 58 samples*  
*Tested on: 12 holdout samples*