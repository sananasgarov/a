import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Import libraries
# (Already imported above)

# 2. Load & explore data
# The file is named a.json but contains CSV formatted data based on inspection
try:
    df = pd.read_csv('a.json')
except:
    # Fallback if read_csv fails on .json extension without specific handling, though typically it reads content
    # We might need to handle it if it really is JSON structure, but the preview showed CSV.
    # Assuming CSV format based on the "id,gender,age..." header.
    pass

print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())

# 3. Data preprocessing / feature engineering

# Handling Missing Values
# 'bmi' column has 'N/A' strings in the preview. Let's force it to numeric and handle NaNs.
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# DROP ID as it's not a feature
df = df.drop('id', axis=1)

# Encoding Categorical Variables
# We will use one-hot encoding or label encoding. For scratch implementation, converting to simple numeric codes is often easier to inspect.
# categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
# Let's use get_dummies for simplicity in preprocessing. Ensure dtype is int/float to avoid boolean columns.
df = pd.get_dummies(df, drop_first=True, dtype=int)

# Feature Scaling (Crucial for Gradient Descent from scratch)
# We will scale all columns except the target 'stroke'
target = 'stroke'
features = [c for c in df.columns if c != target]

X = df[features].values.astype(float) # Ensure all data is float for numpy operations
y = df[target].values


# Normalize features: (x - min) / (max - min) or (x - mean) / std
# Using Standardization (Z-score normalization)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / (std + 1e-8) # Add small epsilon to avoid division by zero

# Add intercept term (column of 1s) to X
X = np.c_[np.ones(X.shape[0]), X]

# 4. Train/Test split
def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    test_samples = int(X.shape[0] * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2)

# 5. Choose model (Logistic Regression from Scratch)
# Since the target 'stroke' is 0 or 1, this is a classification problem.

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for i in range(self.iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights)
            # Activation
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            
            # Compute loss (optional, for monitoring)
            # Binary Cross Entropy
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_predicted, epsilon, 1 - epsilon)
            loss = - (1/n_samples) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.losses.append(loss)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss {loss}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights)
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_predicted_cls = [1 if i > threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)

# 6. Train model
print("\nTraining Logistic Regression from scratch...")
model = LogisticRegressionScratch(learning_rate=0.1, iterations=2000)
model.fit(X_train, y_train)

# Plot Loss Curve
plt.plot(model.losses)
plt.title("Loss over iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
# plt.show() # Uncomment to see the plot

# 7. Predict
print("\nPredicting on test set...")
y_pred = model.predict(X_test)

# 8. Evaluate (Accuracy, Precision, Recall, F1)
def evaluate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
        
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
        
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = evaluate(y_test, y_pred)

print("-" * 30)
print("Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("-" * 30)
