import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
data_set = pd.read_csv("/content/AmericanExpress-Data-Analysis-for-User-Exit-Prediction.csv")

# Features and Labels
X = data_set.iloc[:, :-1]
Y = data_set.iloc[:, -1].values  # Convert to numpy array

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])  # Assuming column name is 'Gender'

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Assuming column name is 'Geography'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Geography'])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # Convert to NumPy array

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define ANN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

input_dim = X_train.shape[1]
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, Y_train, batch_size=32, epochs=55, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Single Prediction
sample_input = sc.transform([[1.0, 0.0, 0.0, 447, 1, 31, 7, 0.0, 4, 1, 519360]])
print((model.predict(sample_input) > 0.5)[0][0])

# Predict on Test Set
Y_pred = model.predict(X_test)
Y_pred_binary = (Y_pred > 0.5)

# Compare predictions and actual values
print(np.concatenate((Y_pred_binary, Y_test.reshape(-1,1)), axis=1))

# Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred_binary)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy Score: {accuracy_score(Y_test, Y_pred_binary):.4f}")
