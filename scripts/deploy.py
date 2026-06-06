import os
from huggingface_hub import HfApi

def deploy():
    # Use token from environment or fallback to the hardcoded token if needed
    hf_token = os.getenv("HF_TOKEN")
    space_repo = os.getenv("HF_SPACE_REPO", "pvinayv/sales-forecast-app")
    
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set.")
        return

    print(f"Initializing Hugging Face Space Deployment to {space_repo}...")
    api = HfApi()

    # Create the Space if it doesn't exist
    try:
        api.create_repo(
            repo_id=space_repo,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=hf_token,
            exist_ok=True
        )
        print("Space repository initialized or already exists.")
    except Exception as e:
        print(f"Space initialization warning: {e}")

    # Files required for Streamlit deployment via Docker
    files_to_upload = ["app.py", "requirements.txt", "Dockerfile"]
    
    for file_name in files_to_upload:
        if os.path.exists(file_name):
            print(f"Uploading {file_name}...")
            try:
                api.upload_file(
                    path_or_fileobj=file_name,
                    path_in_repo=file_name,
                    repo_id=space_repo,
                    repo_type="space",
                    token=hf_token
                )
                print(f"Successfully uploaded {file_name}")
            except Exception as e:
                print(f"Failed to upload {file_name}: {e}")
        else:
            print(f"Warning: {file_name} not found locally.")

    print("Hugging Face Space Deployment completed successfully!")

if __name__ == "__main__":
    deploy()
