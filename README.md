# Project Overview

This project aims to predict housing prices in California using machine learning techniques. The implementation is divided into three key stages:

## 1. Preprocessing
In this stage, the dataset is prepared for training by handling missing or misfit data, encoding categorical variables, and normalizing numerical values. The following steps are performed:

### Encoding
- Categorical text features are processed using `OneHotEncoder` and `OrdinalEncoder`.

### Handling Missing or Misfit Data
Three strategies are available for addressing missing or misfit data:
1. `drop_rows`: Remove rows containing missing or invalid data.
2. `drop_column`: Remove columns containing missing or invalid data.
3. `fill_miss`: Fill missing data using one of the following methods:
   - `median_imputer`: Replace missing values with the median.
   - `knn_imputer`: Use K-Nearest Neighbors imputation.

### Normalization
The data is normalized using one of the following methods:
1. **Min-Max Scaling**: Scale values between 0 and 1, with the minimum and maximum values set as bounds.
2. **Standardization**: Center data to have a mean of 0 and scale values based on their standard deviation.

---

## 2. Training
The preprocessed data is used to train the model using three machine learning algorithms:
1. **Linear Regression**
2. **Decision Trees**
3. **Random Forest**

---

## 3. Evaluation
The trained model is evaluated using the following metrics and outputs:
- **RMSE (Root Mean Square Error)**: Measures the model's prediction accuracy.
- **Covariance Matrix**: Provides insights into relationships between features.
- **Confidence Level**: Indicates the model's reliability.

---

This structured pipeline ensures an effective and transparent approach to predicting housing prices in California. Let us know your feedback or suggestions!
