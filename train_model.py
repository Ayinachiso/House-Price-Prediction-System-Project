import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample dataset (area, bedrooms, bathrooms, stories, parking)
X = np.array([
    [1200, 3, 2, 1, 1],
    [1500, 4, 3, 2, 2],
    [800, 2, 1, 1, 0],
    [2000, 5, 4, 2, 3],
    [1000, 3, 2, 1, 1],
    [1800, 4, 3, 2, 2]
])

# House prices
y = np.array([150000, 220000, 90000, 320000, 140000, 260000])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(5,)))
model.add(Dense(12, activation="relu"))
model.add(Dense(1))

# Compile model
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=4)

# Save model
model.save("model.h5")