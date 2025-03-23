import os
import sys
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
import json

# Load dataset paths from command-line arguments or use default testing paths
if len(sys.argv) == 5:
    healthcare_path = sys.argv[1]
    activity_environment_path = sys.argv[2]
    digital_interaction_path = sys.argv[3]
    personal_health_path = sys.argv[4]
    print("Dataset paths loaded from command-line arguments.")
else:
    print("Dataset paths not provided as arguments, using default paths.")
    healthcare_path = os.path.join('uploads', 'healthcare_dataset.csv')
    activity_environment_path = os.path.join('uploads', 'activity_environment_data.csv')
    digital_interaction_path = os.path.join('uploads', 'digital_interaction_data.csv')
    personal_health_path = os.path.join('uploads', 'personal_health_data.csv')

# Load datasets
personal_health_data = pd.read_csv(personal_health_path)
healthcare_data = pd.read_csv(healthcare_path)
digital_interaction_data = pd.read_csv(digital_interaction_path)
activity_environment_data = pd.read_csv(activity_environment_path)

print("Datasets loaded successfully.")

# Ensure 'User_ID' exists in all datasets, generate if missing and convert to string
datasets = [personal_health_data, healthcare_data, digital_interaction_data, activity_environment_data]
for i, df in enumerate(datasets):
    if 'User_ID' not in df.columns:
        print(f"'User_ID' not found in dataset {i + 1}. Generating synthetic IDs.")
        df['User_ID'] = range(1, len(df) + 1)
    
    # Convert 'User_ID' to string to prevent merge conflicts
    df['User_ID'] = df['User_ID'].astype(str)

print("Ensured 'User_ID' consistency across datasets.")

# Merge datasets on 'User_ID'
merged_data = personal_health_data.merge(healthcare_data, on='User_ID', how='outer') \
    .merge(digital_interaction_data, on='User_ID', how='outer') \
    .merge(activity_environment_data, on='User_ID', how='outer')

print("Datasets merged successfully.")

# Handle missing values only for numeric columns
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())
print("Missing values handled.")

# Encode categorical variables (one-hot encoding) only for categorical columns
categorical_cols = merged_data.select_dtypes(include=['object', 'category']).columns
merged_data = pd.get_dummies(merged_data, columns=categorical_cols, drop_first=True)
print("Categorical variables encoded.")

# Scale numeric features
merged_data[numeric_cols] = StandardScaler().fit_transform(merged_data[numeric_cols])
print("Numeric features scaled.")

# Calculate BMI if height and weight are available
if 'Weight' in merged_data.columns and 'Height' in merged_data.columns:
    merged_data['BMI'] = merged_data['Weight'] / (merged_data['Height'] / 100) ** 2
    print("BMI calculated.")

visualization_dir = os.path.join('static', 'visualizations')
os.makedirs(visualization_dir, exist_ok=True)
print(f"Visualization directory set to: {visualization_dir}")

# Sample data for faster visualizations (e.g., 10% of data)
sampled_data = merged_data.sample(frac=0.1, random_state=42)

# Ensure required columns exist or generate fallback values
if 'Age' not in sampled_data.columns:
    print("Column 'Age' not found. Generating synthetic Age data.")
    sampled_data['Age'] = np.random.randint(20, 70, size=len(sampled_data))

if 'Heart_Rate' not in sampled_data.columns:
    print("Column 'Heart_Rate' not found. Generating synthetic Heart Rate data.")
    sampled_data['Heart_Rate'] = np.random.randint(60, 100, size=len(sampled_data))

if 'Stress_Level' not in sampled_data.columns:
    print("Column 'Stress_Level' not found. Generating synthetic Stress Level data.")
    sampled_data['Stress_Level'] = np.random.randint(1, 10, size=len(sampled_data))

if 'Sleep_Duration' not in sampled_data.columns:
    print("Column 'Sleep_Duration' not found. Generating synthetic Sleep Duration data.")
    sampled_data['Sleep_Duration'] = np.random.uniform(4, 10, size=len(sampled_data))

if 'risk_level' not in sampled_data.columns:
    print("Column 'risk_level' not found. Generating synthetic risk levels.")
    sampled_data['risk_level'] = np.random.choice(['Low', 'Medium', 'High'], size=len(sampled_data))

numeric_cols = sampled_data.select_dtypes(include=[np.number]).columns

# Optimized Correlation Heatmap
try:
    print("Generating Optimized Correlation Heatmap...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(sampled_data[numeric_cols].corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(visualization_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Optimized Correlation Heatmap saved.")
except Exception as e:
    print(f"Error generating Correlation Heatmap: {e}")

# Optimized Heart Rate vs Age plot
try:
    print("Generating Optimized Heart Rate vs Age plot...")
    if 'Age' in sampled_data.columns and 'Heart_Rate' in sampled_data.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Age', y='Heart_Rate', data=sampled_data)
        plt.title('Heart Rate vs Age')
        plt.savefig(os.path.join(visualization_dir, 'heart_rate_vs_age.png'))
        plt.close()
        print("Optimized Heart Rate vs Age plot saved.")
except Exception as e:
    print(f"Error generating Heart Rate vs Age plot: {e}")

# Optimized Stress Level vs Sleep Duration plot
try:
    print("Generating Optimized Stress Level vs Sleep Duration plot...")
    if 'Stress_Level' in sampled_data.columns and 'Sleep_Duration' in sampled_data.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Sleep_Duration', y='Stress_Level', data=sampled_data)
        plt.title('Stress Level vs Sleep Duration')
        plt.savefig(os.path.join(visualization_dir, 'stress_vs_sleep.png'))
        plt.close()
        print("Optimized Stress Level vs Sleep Duration plot saved.")
except Exception as e:
    print(f"Error generating Stress Level vs Sleep Duration plot: {e}")

# Optimized Steps vs Calories Burned plot
try:
    print("Generating Optimized Steps vs Calories Burned plot...")
    if 'Steps' in sampled_data.columns and 'Calories_Burned' in sampled_data.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Steps', y='Calories_Burned', data=sampled_data)
        plt.title('Steps vs Calories Burned')
        plt.savefig(os.path.join(visualization_dir, 'steps_vs_calories.png'))
        plt.close()
        print("Optimized Steps vs Calories Burned plot saved.")
except Exception as e:
    print(f"Error generating Steps vs Calories Burned plot: {e}")

# Optimized Risk Level Distribution plot
try:
    print("Generating Optimized Risk Level Distribution plot...")
    if 'risk_level' in sampled_data.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='risk_level', data=sampled_data, palette='coolwarm')
        plt.title('Risk Level Distribution')
        plt.savefig(os.path.join(visualization_dir, 'risk_level_distribution.png'))
        plt.close()
        print("Optimized Risk Level Distribution plot saved.")
except Exception as e:
    print(f"Error generating Risk Level Distribution plot: {e}")

# Collect visualization paths
visualization_paths = [
    os.path.join(visualization_dir, 'correlation_heatmap.png'),
    os.path.join(visualization_dir, 'heart_rate_vs_age.png'),
    os.path.join(visualization_dir, 'stress_vs_sleep.png'),
    os.path.join(visualization_dir, 'steps_vs_calories.png'),
    os.path.join(visualization_dir, 'risk_level_distribution.png')
]

# Generate placeholder PNGs for visualizations that weren't created
for path in visualization_paths:
    if not os.path.exists(path):
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Visualization Not Generated', fontsize=16, ha='center', va='center')
        plt.axis('off')
        plt.savefig(path)
        plt.close()
        print(f"Placeholder saved for missing visualization: {path}")
