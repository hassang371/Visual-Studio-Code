import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

# Set up visualization directory
visualization_dir = '/Users/hassangameryt/Documents/VS CODE/PYTHON/PY/static/visualizations'
os.makedirs(visualization_dir, exist_ok=True)

# Load health data with improved error handling
health_path = '/Users/hassangameryt/Documents/VS CODE/healthcare_dataset.csv'
try:
    health_data = pd.read_csv(health_path)
except Exception as e:
    print(f"Error loading healthcare_dataset.csv: {e}. Generating synthetic health data...")
    health_data = pd.DataFrame({
        'age': np.random.randint(20, 80, 100),
        'bmi': np.random.uniform(18, 35, 100),
        'blood_pressure': np.random.randint(90, 180, 100),
        'heart_rate': np.random.randint(60, 100, 100),
        'cholesterol': np.random.randint(150, 300, 100),
        'risk_level': np.random.choice([0, 1], 100)
    })

# Load wearable datasets (handle missing files) with improved error handling
wearable_files = [
    '/Users/hassangameryt/Documents/VS CODE/activity_environment_data.csv',
    '/Users/hassangameryt/Documents/VS CODE/digital_interaction_data.csv',
    '/Users/hassangameryt/Documents/VS CODE/personal_health_data.csv'
]
wearable_data_list = []
for f in wearable_files:
    try:
        for chunk in pd.read_csv(f, chunksize=1000):
            wearable_data_list.append(chunk)
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not wearable_data_list:
    print("No wearable data files found. Proceeding with health data only.")
    wearable_data = pd.DataFrame()
else:
    wearable_data = pd.concat(wearable_data_list, axis=0, ignore_index=True)
    wearable_data = wearable_data.apply(pd.to_numeric, errors='coerce')
    wearable_data = wearable_data.fillna(wearable_data.median())

# Preprocess wearable data: Select only numeric columns and apply median imputation
wearable_numeric_cols = wearable_data.select_dtypes(include=[np.number]).columns

if not wearable_numeric_cols.empty:
    wearable_data = wearable_data.reset_index(drop=True)
    wearable_data[wearable_numeric_cols] = wearable_data[wearable_numeric_cols].fillna(wearable_data[wearable_numeric_cols].median())
else:
    print("No numeric columns found in wearable data. Skipping median imputation.")

# Health Data Preprocessing: Select only numeric columns and fill missing values
# Encode categorical columns and handle non-numeric data
categorical_cols = health_data.select_dtypes(include=['object', 'category']).columns
health_data = pd.get_dummies(health_data, columns=categorical_cols, drop_first=True)
numeric_cols = health_data.select_dtypes(include=[np.number]).columns
health_data[numeric_cols] = health_data[numeric_cols].fillna(health_data[numeric_cols].median())
X = health_data[numeric_cols]

# Handle risk_level column or generate synthetic labels
if 'risk_level' in health_data.columns:
    y = health_data['risk_level']
    X = health_data.drop('risk_level', axis=1)
else:
    print("Warning: 'risk_level' column not found. Skipping label generation.")
    y = np.zeros(len(health_data))

# Scale health data
scaler = StandardScaler()
scaled_health_data = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_health_data, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Calculate feature importances after training the RandomForest model
feature_importances = rf_model.feature_importances_

# Generate Random Forest Risk Scores
rf_probs = rf_model.predict_proba(X_test)
if rf_probs.shape[1] == 1:
    rf_probs = np.zeros(len(rf_probs))
else:
    rf_probs = rf_probs[:, 1]
rf_risk_scores = ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in rf_probs]

# Early stopping to prevent overfitting during LSTM training
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

sequence_length = 10

def train_lstm_model(X_train, y_train, X_test, y_test):
    model = models.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.LSTM(50, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model

# Train LSTM for wearable data
trimmed_length = (len(wearable_data) // sequence_length) * sequence_length
wearable_data = wearable_data.iloc[:trimmed_length]
n_features = wearable_data.shape[1]

X_wearable = np.array(wearable_data).reshape(-1, sequence_length, n_features)
y_wearable = np.random.choice([0, 1], len(X_wearable))

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_wearable, y_wearable, test_size=0.2, random_state=42)

lstm_model = train_lstm_model(X_w_train, y_w_train, X_w_test, y_w_test)

lstm_probs = lstm_model.predict(X_w_test).flatten()
lstm_risk_scores = ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in lstm_probs]

# Enhanced Risk Scoring
def enhanced_risk_score(row):
    score = 0
    score += row.get('Steps', 0) < 3000
    score += row.get('Sleep_Duration', 0) < 5
    score += row.get('Calories_Burned', 0) < 2000
    return 'High' if score >= 2 else 'Medium' if score == 1 else 'Low'

wearable_data['enhanced_risk'] = wearable_data.apply(enhanced_risk_score, axis=1)

# Combined Risk Scores
def combine_risk_scores(rf_scores, lstm_scores):
    return [
        'High' if rf == 'High' or lstm == 'High' else 'Medium' if rf == 'Medium' or lstm == 'Medium' else 'Low'
        for rf, lstm in zip(rf_scores, lstm_scores)
    ]

combined_risk_scores = combine_risk_scores(rf_risk_scores, lstm_risk_scores)

# Create a DataFrame with combined risk scores for the test set
combined_risk_df = pd.DataFrame({'combined_risk': combined_risk_scores})

# Add combined risk scores only to the test set portion
wearable_data.loc[wearable_data.index[-len(combined_risk_df):], 'combined_risk'] = combined_risk_df.values.flatten()

# Improved visualizations with descriptive labels and enhanced color schemes
def create_visualization(data, title, filename, kind='bar'):
    plt.figure(figsize=(10, 6))
    if kind == 'heatmap':
        numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
        sns.heatmap(numeric_data.corr(), annot=False, cmap='coolwarm')
        plt.xlabel('Features')
        plt.ylabel('Features')
    elif kind == 'countplot':
        sns.countplot(x='combined_risk', data=data, palette='coolwarm')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
    elif kind == 'barplot':
        if len(feature_importances) == len(X.columns):
            sns.barplot(x=feature_importances, y=X.columns)
        else:
            print("Warning: Feature importances do not match number of features.")
    plt.title(title)
    plt.savefig(os.path.join(visualization_dir, filename))
    plt.close()

create_visualization(wearable_data, 'Wearable Data Correlation Heatmap', 'wearable_data_heatmap.png', 'heatmap')
create_visualization(wearable_data, 'Combined Risk Score Distribution', 'combined_risk_distribution.png', 'countplot')
create_visualization(wearable_data, 'Feature Importance - Random Forest', 'feature_importance.png', 'barplot')

# Print reports
print("Random Forest Report:\n", classification_report(y_test, rf_preds))
print("Wearable Data Columns:", wearable_data.columns) 