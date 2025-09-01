# Version 2 - Performance Results

## Optimal Parameters
- Polynomial Degree: 8
- Regularization Parameter: λ = 0.030
- Training Iterations: 200
- Final Cost: 11,223.34

## Prediction Examples
| Size (m²) | Rooms | Predicted Rent |
|-----------|-------|----------------|
| 68 | 2 | €811.13 |
| 85 | 3 | €924.00 |
| 100 | 4 | €968.94 |
| 120 | 4 | €1069.48 |
| 55 | 1 | €738.13 |

## Performance Metrics
- **Test Error**: 13,520.55
- **Mean Absolute Error**: €131.34
- **R-squared**: 0.8207

## Comparison with Version 1
- **Error Reduction**: 4.6% improvement in MAE
- **Explanatory Power**: 25.8% improvement in R²
- **Advanced Features**: Polynomial regression + regularization

## Technical Validation
- Learning curves show effective training
- Validation curves confirm optimal parameter selection
- Cost convergence demonstrates stable optimization