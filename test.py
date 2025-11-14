from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = fetch_olivetti_faces()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Load saved model
model = joblib.load("savedmodel.pth")

# Evaluate
accuracy = model.score(X_test, y_test)
print("Test accuracy:", accuracy)
