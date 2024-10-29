import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Sample DataFrame creation for the business use case
data = {
    'Job_Satisfaction': [3, 2, 4, 1, 2, 4, 3, 1, 5, 2],
    'Num_Projects': [5, 3, 7, 2, 3, 5, 6, 2, 4, 3],
    'Working_Hours_Per_Week': [45, 40, 50, 35, 40, 48, 47, 32, 45, 38],
    'Years_at_Company': [2, 5, 3, 10, 4, 3, 7, 1, 5, 3],
    'Attrition': [1, 0, 1, 0, 0, 1, 1, 0, 0, 1]  # 1 for likely to leave, 0 for likely to stay
}

# Creating the DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df[['Job_Satisfaction', 'Num_Projects', 'Working_Hours_Per_Week', 'Years_at_Company']]
y = df['Attrition']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Defining the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), ['Job_Satisfaction', 'Num_Projects', 'Working_Hours_Per_Week', 'Years_at_Company'])
    ]
)

# Building the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# Training the pipeline
pipeline.fit(X_train, y_train)

# Making predictions
y_pred = pipeline.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Pipeline Accuracy:", accuracy)
print("Pipeline Classification Report:\n", report)
