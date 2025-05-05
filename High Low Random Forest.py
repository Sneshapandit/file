"""To prioritize high-value listings, a real estate firm wants to predict whether a house falls
in the "High" or "Low" price category. Build a Random Forest classifier to perform binary
classification based on the house’s features"""
# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the dataset
df = pd.read_csv('USA_Housing.csv')

# 3. Drop unnecessary columns
df = df.drop('Address', axis=1)

# 4. Create binary target column: 'High' if Price > median, else 'Low'
median_price = df['Price'].median()
df['Price_Category'] = np.where(df['Price'] > median_price, 'High', 'Low')

# 5. Features and target
X = df.drop(['Price', 'Price_Category'], axis=1)
y = df['Price_Category']

# 6. Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'High' -> 1, 'Low' -> 0

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 8. Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Visualize Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - House Price Category")
plt.show()
