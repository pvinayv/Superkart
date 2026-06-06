import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# 1. Ensure raw dataset is registered on Hugging Face first
# If running locally and we have the raw file, make sure it's uploaded to Hugging Face
hf_token = os.getenv("HF_TOKEN")
hf_repo = os.getenv("HF_DATASET_REPO", "pvinayv/superkart-dataset")

raw_hf_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/data/SuperKart.csv"

try:
    print(f"Attempting to load raw dataset directly from Hugging Face: {raw_hf_url}")
    df = pd.read_csv(raw_hf_url)
    print("Raw dataset loaded directly from Hugging Face successfully.")
except Exception as e:
    print(f"Could not load directly from Hugging Face: {e}")
    local_path = 'data/SuperKart.csv'
    if os.path.exists(local_path):
        print("Registering local dataset to Hugging Face dataset space...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=hf_repo, repo_type="dataset", token=hf_token, exist_ok=True)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo='data/SuperKart.csv',
                repo_id=hf_repo,
                repo_type='dataset',
                token=hf_token
            )
            print("Successfully registered raw data on Hugging Face dataset space.")
            df = pd.read_csv(raw_hf_url)
        except Exception as upload_err:
            print(f"Failed to register dataset on Hugging Face: {upload_err}")
            print("Falling back to local raw data.")
            df = pd.read_csv(local_path)
    else:
        print("Error: Local data/SuperKart.csv not found and cannot load from Hugging Face.")
        exit(1)

# 2. Perform Data Cleaning
# Handling missing values
df['Product_Weight'] = df['Product_Weight'].fillna(df['Product_Weight'].mean())
df['Store_Size'] = df['Store_Size'].fillna('Medium')
# Unify Product Sugar Content categories
df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({
    'low fat': 'Low Fat', 
    'LF': 'Low Fat', 
    'reg': 'Regular'
})
# Remove any unnecessary columns if needed (e.g. none specified, keeping all for now)
print("Data cleaning completed.")

# 3. Split into train and test
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save locally
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
print("Train and Test splits saved locally in data/ folder.")

# 4. Upload to Hugging Face
hf_token = os.getenv("HF_TOKEN")
hf_repo = os.getenv("HF_DATASET_REPO", "pvinayv/superkart-dataset") # e.g. user/superkart-dataset

if hf_token:
    try:
        api = HfApi()
        print(f"Uploading splits to HF Dataset: {hf_repo}")
        api.upload_file(
            path_or_fileobj='data/train.csv',
            path_in_repo='train.csv',
            repo_id=hf_repo,
            repo_type='dataset',
            token=hf_token
        )
        api.upload_file(
            path_or_fileobj='data/test.csv',
            path_in_repo='test.csv',
            repo_id=hf_repo,
            repo_type='dataset',
            token=hf_token
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
else:
    print("HF_TOKEN or valid HF_DATASET_REPO not set. Skipping HF upload.")
