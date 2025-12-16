# Insurance Cost Prediction: Feature Selection Analysis - Comprehensive Explanation

## Dataset Overview

The Insurance Cost Prediction dataset contains information about healthcare insurance customers and their associated premiums. It consists of 7 columns:

- **Age**: The insured person's age.
- **Sex**: Gender (male or female) of the insured.
- **BMI (Body Mass Index)**: A measure of body fat based on height and weight.
- **Children**: The number of dependents covered.
- **Smoker**: Whether the insured is a smoker (yes or no).
- **Region**: The geographic area of coverage.
- **Charges**: The medical insurance costs incurred by the insured person.

This dataset is ideal for regression analysis to predict insurance costs based on these features.

## Objectives of the Analysis

The primary goals of this analysis are:

1. **Predict Insurance Charges**: Use Linear Regression and K-Nearest Neighbors (KNN) regression to predict insurance charges.
2. **Feature Selection**: Demonstrate how removing weak features (those with little correlation to the target) can maintain or even improve model efficiency without sacrificing accuracy.
3. **Model Comparison**: Compare baseline models (using all features) to optimized models (using selected features).

## Data Preprocessing

The preprocessing steps are essential for preparing the data for machine learning models:

### 1. Data Loading and Initial Inspection
- Load the CSV data
- Examine the data structure
- Check for missing values and data types

### 2. Data Type Conversion
- Convert numeric columns (age, bmi, children, charges) to float
- Convert categorical columns (sex, smoker, region) to category type

### 3. Handling Categorical Variables
- Apply One-Hot Encoding to convert categorical variables to numeric
- Use `drop_first=True` to avoid multicollinearity (dummy variable trap)
- This creates binary variables for each category except one reference category

### 4. Feature and Target Separation
- Define X (Features): All columns except 'charges'
- Define y (Target): 'charges' column

## Model Implementation

### 1. Linear Regression
Linear regression models the relationship between the target variable and features using a linear equation. The model finds the best-fitting line through the data points.

**Advantages:**
- Simple to understand and implement
- Fast training time
- Provides interpretable coefficients
- Less prone to overfitting

**Disadvantages:**
- Assumes linear relationship between features and target
- Sensitive to outliers
- Requires feature scaling for optimal performance

### 2. K-Nearest Neighbors (KNN) Regression
KNN regression predicts the target value based on the average of the k-nearest neighbors in the feature space.

**Advantages:**
- No assumption about data distribution
- Simple and intuitive
- Works well with small datasets
- Can capture non-linear relationships

**Disadvantages:**
- Computationally expensive during prediction
- Sensitive to feature scaling
- Performance degrades with high-dimensionality
- Requires more memory

## Feature Selection Process

### 1. Correlation Analysis
- Calculate correlation matrix to identify relationships between features
- Focus on correlations with the target variable 'charges'
- Identify weak features that have minimal correlation with the target

### 2. Correlation Findings
Based on the analysis, the correlation with charges was found as follows:
- `smoker_yes`: Very high correlation (> 0.78) - most important feature
- `age`: Moderate correlation (~0.3)
- `bmi`: Moderate correlation (~0.2)
- `children`: Low correlation (~0.067)
- `sex_male`: Very low, near zero (~0.057)
- `region_*`: All region variables have correlations very close to zero

### 3. Selected Features Decision
The relationship between `sex` and `region` with `charges` is very weak. Including them might add noise and complexity without adding predictive value. Therefore, these features are removed to optimize the model.

## Model Training Process

### 1. Data Splitting
- Split data into 80% training and 20% testing
- Use fixed random_state for reproducibility
- Ensure both splits maintain the same data distribution

### 2. Feature Scaling
- Apply StandardScaler to normalize features
- Scale features to have mean=0 and variance=1
- Critical for KNN as it's distance-based

### 3. Model Training
- Train both Linear Regression and KNN models
- Use the same train/test split for fair comparison
- Evaluate both the full model (all features) and the selected model

## Model Evaluation Metrics

### 1. R² Score (Coefficient of Determination)
- Measures the proportion of variance in the target variable explained by the model
- Ranges from 0 to 1 (higher is better)
- A value of 1 indicates perfect prediction

### 2. Mean Squared Error (MSE)
- Average of the squared differences between predicted and actual values
- Lower values indicate better performance
- Sensitive to outliers due to squaring

## Results Analysis

### 1. Baseline Model (All Features) Results:
- Linear Regression: R² Score ~0.7836, MSE ~33,596,916
- KNN: R² Score ~0.8038, MSE ~30,459,866

### 2. Optimized Model (Selected Features) Results:
- Linear Regression: R² Score ~0.7811, MSE ~33,981,654
- KNN: R² Score ~0.8693, MSE ~20,286,205

## Conclusions

### 1. Efficiency
By removing `sex` and `region`, the number of features was reduced. For large datasets, this significantly reduces training time and memory usage.

### 2. Accuracy
- The accuracy (R² Score) remained almost effectively the same (or dropped extremely negligibly) after removing the weak features.
- This proves the hypothesis: `sex` and `region` are not important predictors for insurance charges in this dataset.
- KNN showed significant improvement with the reduced feature set.

### 3. Model Logic
The model is now cleaner and easier to interpret, focusing only on the drivers that matter: Age, BMI, Smoking, and Children.

### 4. Key Insights
- **Smoking is the strongest predictor** of insurance charges, which is consistent with actuarial principles.
- **Age** has a moderate positive correlation with charges.
- **BMI** also shows moderate correlation, indicating health risk assessment.
- **Children** has a weak but positive correlation, possibly indicating family coverage.
- Geographic **region** surprisingly shows minimal correlation with insurance costs in this dataset.

## Technical Implementation Details

1. **Data Preprocessing**: One-hot encoding was used to convert categorical variables into numeric format for the algorithms.
2. **Feature Scaling**: Applied to ensure all features contribute equally to the model, especially important for KNN.
3. **Cross-Validation**: Consistent train/test split used for all comparisons to ensure fair evaluation.
4. **Hyperparameter Tuning**: Basic KNN with k=5 used as a starting point, though optimal k value could be explored.

This analysis demonstrates effective feature selection can improve model performance while reducing computational complexity, making models more efficient and interpretable.