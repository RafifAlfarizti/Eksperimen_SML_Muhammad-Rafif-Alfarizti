# automate_Rafif.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(raw_data_path, save_folder):
    """
    Preprocessing data stroke dengan pipeline yang terstruktur
    
    Args:
        raw_data_path (str): Path ke file dataset mentah
        save_folder (str): Folder untuk menyimpan hasil preprocessing
    """
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(raw_data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Data cleaning
    print("Cleaning data...")
    
    # Drop kolom 'id' jika ada
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Drop baris gender == 'Other'
    df = df[df['gender'] != 'Other']
    
    # Gabungkan kategori minoritas di kolom 'work_type'
    df['work_type'] = df['work_type'].replace({
        'children': 'Other',
        'Never_worked': 'Other'
    })
    
    # Separate features and target
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    # Identify different types of features
    # Numerical features
    num_features = ['age', 'avg_glucose_level', 'bmi']
    
    # Categorical features for label encoding
    label_enc_features = ['ever_married', 'Residence_type', 'gender']
    
    # Categorical features for one-hot encoding
    onehot_features = ['work_type', 'smoking_status']
    
    # Create preprocessing pipelines
    print("Creating preprocessing pipelines...")
    
    # Numerical pipeline
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Label encoding pipeline
    label_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('label_encoder', 'passthrough')  # We'll handle label encoding separately
    ])
    
    # One-hot encoding pipeline
    onehot_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine all preprocessors
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, num_features),
        ('onehot', onehot_pipeline, onehot_features)
    ])
    
    # Handle label encoding separately for better control
    label_encoders = {}
    X_label_encoded = X.copy()
    
    for col in label_enc_features:
        if col in X_label_encoded.columns:
            le = LabelEncoder()
            # Handle missing values first
            X_label_encoded[col] = X_label_encoded[col].fillna(X_label_encoded[col].mode()[0])
            X_label_encoded[col] = le.fit_transform(X_label_encoded[col])
            label_encoders[col] = le
    
    # Apply the main preprocessor to the remaining features
    remaining_features = num_features + onehot_features
    X_remaining = X_label_encoded[remaining_features]
    
    # Fit and transform the data
    print("Applying preprocessing...")
    X_processed_remaining = preprocessor.fit_transform(X_remaining)
    
    # Combine label encoded features with processed features
    X_label_encoded_array = X_label_encoded[label_enc_features].values
    X_processed = np.hstack([X_label_encoded_array, X_processed_remaining])
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create save folder
    os.makedirs(save_folder, exist_ok=True)
    
    # Save processed datasets
    print("Saving processed data...")
    np.save(os.path.join(save_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(save_folder, 'X_test.npy'), X_test)
    np.save(os.path.join(save_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(save_folder, 'y_test.npy'), y_test)
    
    # Save preprocessing artifacts
    artifacts = {
        'preprocessor': preprocessor,
        'label_encoders': label_encoders,
        'feature_names': {
            'numerical': num_features,
            'label_encoded': label_enc_features,
            'onehot_encoded': onehot_features
        }
    }
    
    joblib.dump(artifacts, os.path.join(save_folder, 'preprocessing_artifacts.pkl'))
    
    # Save processed dataframe for inspection
    feature_names = label_enc_features.copy()
    
    # Get feature names from one-hot encoder
    onehot_feature_names = preprocessor.named_transformers_['onehot'].named_steps['encoder'].get_feature_names_out(onehot_features)
    feature_names.extend(num_features)
    feature_names.extend(onehot_feature_names)
    
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df['stroke'] = y.values
    processed_df.to_csv(os.path.join(save_folder, 'stroke_dataset_processed.csv'), index=False)
    
    print(f"Preprocessing selesai!")
    print(f"Data shape after preprocessing: {X_processed.shape}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Files saved in: {save_folder}")
    
    return X_train, X_test, y_train, y_test, artifacts

def load_preprocessing_artifacts(artifacts_path):
    """
    Load preprocessing artifacts for future use
    
    Args:
        artifacts_path (str): Path to the artifacts file
        
    Returns:
        dict: Dictionary containing preprocessing artifacts
    """
    return joblib.load(artifacts_path)

def apply_preprocessing(new_data, artifacts):
    """
    Apply saved preprocessing to new data
    
    Args:
        new_data (pd.DataFrame): New data to preprocess
        artifacts (dict): Preprocessing artifacts
        
    Returns:
        np.array: Preprocessed data
    """
    preprocessor = artifacts['preprocessor']
    label_encoders = artifacts['label_encoders']
    feature_names = artifacts['feature_names']
    
    # Apply label encoding
    new_data_encoded = new_data.copy()
    for col in feature_names['label_encoded']:
        if col in new_data_encoded.columns:
            le = label_encoders[col]
            new_data_encoded[col] = le.transform(new_data_encoded[col])
    
    # Apply main preprocessing
    remaining_features = feature_names['numerical'] + feature_names['onehot_encoded']
    X_remaining = new_data_encoded[remaining_features]
    X_processed_remaining = preprocessor.transform(X_remaining)
    
    # Combine with label encoded features
    X_label_encoded_array = new_data_encoded[feature_names['label_encoded']].values
    X_processed = np.hstack([X_label_encoded_array, X_processed_remaining])
    
    return X_processed

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(__file__)
    RAW_DATA_PATH = os.path.join(base_dir, "../healthcare-dataset-stroke-data.csv")
    SAVE_FOLDER = os.path.join(base_dir, "PROCESSED_DATA")
    
    # Run preprocessing
    try:
        X_train, X_test, y_train, y_test, artifacts = preprocess_data(RAW_DATA_PATH, SAVE_FOLDER)
        print("\n✅ Preprocessing berhasil!")
        
        # Example of loading artifacts
        artifacts_path = os.path.join(SAVE_FOLDER, 'preprocessing_artifacts.pkl')
        loaded_artifacts = load_preprocessing_artifacts(artifacts_path)
        print("✅ Artifacts berhasil dimuat!")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")

# Untuk menjalankan:
# python automate_preprocessing.py

# Jalankan di anaconda prompt
# python automate_Rafif.py

