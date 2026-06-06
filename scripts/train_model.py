import pandas as pd
import os
# pyrefly: ignore [missing-import]
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from huggingface_hub import HfApi

# 1. Load train and test data directly from Hugging Face Data Space
hf_repo_dataset = os.getenv("HF_DATASET_REPO", "pvinayv/superkart-dataset")
train_hf_url = f"https://huggingface.co/datasets/{hf_repo_dataset}/resolve/main/train.csv"
test_hf_url = f"https://huggingface.co/datasets/{hf_repo_dataset}/resolve/main/test.csv"

print(f"Attempting to load train dataset directly from Hugging Face: {train_hf_url}")
try:
    train = pd.read_csv(train_hf_url)
    print("Train dataset loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Could not load train dataset from Hugging Face: {e}. Falling back to local data/train.csv")
    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        print("Train data not found locally. Ensure data_preparation.py has been run.")
        exit(1)
    train = pd.read_csv(train_path)

print(f"Attempting to load test dataset directly from Hugging Face: {test_hf_url}")
try:
    test = pd.read_csv(test_hf_url)
    print("Test dataset loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Could not load test dataset from Hugging Face: {e}. Falling back to local data/test.csv")
    test_path = 'data/test.csv'
    if not os.path.exists(test_path):
        print("Test data not found locally. Ensure data_preparation.py has been run.")
        exit(1)
    test = pd.read_csv(test_path)

X_train = train.drop(columns=['Product_Store_Sales_Total', 'Product_Id', 'Store_Id'])
y_train = train['Product_Store_Sales_Total']
X_test = test.drop(columns=['Product_Store_Sales_Total', 'Product_Id', 'Store_Id'])
y_test = test['Product_Store_Sales_Total']

# 2. Define Model and Parameters
categorical_features = ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Location_City_Type', 'Store_Type']
numeric_features = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Establishment_Year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5,
    random_state=42
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# 3. Tune and Train
print("Training XGBoost Model...")
pipeline.fit(X_train, y_train)

# 4. Evaluate Performance
y_pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation - RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Save best model
joblib.dump(pipeline, 'xgboost_model.pkl')
print("Model saved to xgboost_model.pkl")

# 5. Register in Hugging Face Model Hub
hf_token = os.getenv("HF_TOKEN")
hf_repo = os.getenv("HF_MODEL_REPO", "pvinayv/superkart-model")

if hf_token:
    try:
        api = HfApi()
        print(f"Uploading model to HF Model Hub: {hf_repo}")
        api.upload_file(
            path_or_fileobj='xgboost_model.pkl',
            path_in_repo='xgboost_model.pkl',
            repo_id=hf_repo,
            repo_type='model',
            token=hf_token
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Error uploading to Hugging Face Model Hub: {e}")
else:
    print("HF_TOKEN or valid HF_MODEL_REPO not set. Skipping HF model upload.")
