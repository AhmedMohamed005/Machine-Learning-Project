# Fashion-MNIST Classification Analysis: Logistic Regression and K-Means Clustering

## Introduction

Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

This analysis performs classification on the Fashion-MNIST dataset using two different approaches:
1. **Logistic Regression (Supervised Learning)**
2. **K-Means Clustering (Unsupervised Learning)**

## Dataset Information

- **Source**: Zalando's article images
- **Format**: 28x28 grayscale images
- **Classes**: 10 distinct clothing categories
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Classes in Analysis**: For this study, we focus on 5 classes:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat

## Methodology

### Data Loading and Preprocessing

The dataset is loaded from local `.gz` ubyte files using custom functions:
- `load_mnist_images()`: Loads the image data from compressed files
- `load_mnist_labels()`: Loads the corresponding labels

The data preprocessing pipeline includes:
1. **Normalization**: Pixel values normalized from 0-255 to 0-1 range
2. **Filtering**: Restricting to first 5 classes for this analysis
3. **Data Splitting**: The combined dataset is split into:
   - Training: 24,000 samples
   - Validation: 6,000 samples
   - Test: 5,000 samples

### Dimensionality Reduction with PCA

The original Fashion-MNIST images (28x28) result in 784-dimensional vectors. To reduce computational complexity and improve model performance:

- **Principal Component Analysis (PCA)** is applied
- **Dimensions reduced from 784 to 100**
- **Variance explained**: ~93.6%
- This significant dimensionality reduction preserves most of the relevant information while reducing computational load

### Model 1: Logistic Regression

#### Algorithm Overview
Logistic Regression is a supervised learning algorithm commonly used for classification tasks. Despite its name, it's a linear model for binary classification that can be extended for multinomial classification.

#### Implementation Details
- **Classifier Type**: Multinomial Logistic Regression (for multi-class problem)
- **Regularization**: L2 regularization implemented using the `C` parameter (inverse of regularization strength)
- **Solver**: Both 'lbfgs' and 'saga' options evaluated via grid search
- **Hyperparameter Tuning**: Grid search cross-validation with 5-fold CV
- **Parameters Tested**:
  - C: [0.1, 1, 10]
  - Solver: ['lbfgs', 'saga']

#### Training Process
- Trained on PCA-reduced features
- Fit using training data
- Cross-validation used for hyperparameter selection
- Final model trained on combined training and validation sets

### Model 2: K-Means Clustering

#### Algorithm Overview
K-Means is an unsupervised learning algorithm that partitions n observations into k clusters. Unlike Logistic Regression, K-Means doesn't use label information during training.

#### Implementation Details
- **Number of Clusters**: 5 (matching the number of classes)
- **Initialization**: 'k-means++' for better initialization
- **Max Iterations**: 300
- **Number of Runs**: 10, taking the best result
- **Algorithm**: Lloyd's algorithm

#### Training and Prediction Process
Since K-Means is unsupervised:
- **Training**: K-Means learns cluster centers from training data (without using labels)
- **Label Mapping**: After clustering, each cluster is mapped to the most frequent class in that cluster
- **Prediction**: For test samples, assign to nearest cluster center then use the cluster-to-class mapping

## Results

### Logistic Regression Results
- **Test Accuracy**: ~88.48%
- **Best Parameters**: C=1, solver='lbfgs'
- **Cross-Validation Accuracy**: ~88.59%

#### Per-Class Performance (Logistic Regression):
- T-shirt/top: 91% precision, 91% recall
- Trouser: 98% precision, 96% recall
- Pullover: 84% precision, 80% recall
- Dress: 86% precision, 88% recall
- Coat: 83% precision, 87% recall

### K-Means Clustering Results
- **Test Accuracy**: ~55.72%
- **Significantly lower accuracy compared to Logistic Regression**

#### Per-Class Performance (K-Means):
- T-shirt/top: 96% precision, 51% recall
- Trouser: 73% precision, 86% recall
- Pullover: 28% precision, 32% recall
- Dress: 55% precision, 45% recall
- Coat: 48% precision, 66% recall

## Model Comparison

| Model | Test Accuracy | Type |
|-------|---------------|------|
| Logistic Regression | 0.8848 | Supervised |
| K-Means | 0.5572 | Unsupervised |

### Key Differences Observed:

1. **Supervised vs Unsupervised**:
   - Logistic Regression uses label information during training
   - K-Means learns patterns without knowing the true labels

2. **Performance Gap**:
   - Logistic Regression significantly outperforms K-Means
   - The gap of ~32.76% indicates the importance of label information

3. **Feature Learning**:
   - Logistic Regression learns discriminative boundaries between classes
   - K-Means learns representative prototypes that may not align with class boundaries

## Technical Implementation Details

### Principal Component Analysis (PCA)
- Reduces 784 features (28×28 pixels) to 100 principal components
- Preserves 93.6% of variance in the data
- Significantly reduces computational complexity while maintaining most information

### Model Evaluation Metrics
- **Accuracy**: Overall percentage of correctly classified samples
- **Precision**: Proportion of predicted positives that were correctly identified
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC Curves**: For evaluating multi-class classification performance

### Cross-Validation
- 5-fold cross-validation used for hyperparameter selection
- Ensures robust model selection by testing on multiple data splits

## Conclusion

### Key Findings:
1. **Superior Performance**: Logistic Regression significantly outperformed K-Means clustering
2. **Importance of Labels**: The substantial difference in performance highlights the importance of supervised learning when label information is available
3. **Feature Engineering**: PCA successfully reduced dimensionality while preserving most information
4. **Class Imbalance**: Some classes like 'Trouser' were easier to classify than others like 'Pullover'

### Practical Implications:
- For image classification tasks, supervised learning methods like Logistic Regression typically outperform unsupervised methods
- K-Means clustering is not well-suited for direct classification of image data
- Proper dimensionality reduction techniques like PCA can be highly beneficial

### Limitations:
- Only 5 out of 10 original Fashion-MNIST classes were analyzed
- Simple Logistic Regression without deep learning techniques
- K-Means clustering assumes spherical clusters which may not reflect the true data structure

### Future Improvements:
- Explore more sophisticated supervised learning models (Deep Neural Networks, SVM)
- Investigate ensemble methods to improve performance
- Consider other clustering techniques for potential label discovery
- Expand to all 10 classes of Fashion-MNIST
- Implement more advanced feature extraction techniques

This analysis demonstrates the fundamental differences between supervised and unsupervised learning approaches for image classification, showing that when label information is available, supervised methods like Logistic Regression are more effective than unsupervised methods like K-Means clustering.