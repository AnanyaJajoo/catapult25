import kagglehub

# Download latest version
path = kagglehub.dataset_download("jonathannield/cctv-action-recognition-dataset")

print("Path to dataset files:", path)