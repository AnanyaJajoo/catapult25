import kagglehub

## this is the dataset 

# Download latest version
path = kagglehub.dataset_download("jonathannield/cctv-action-recognition-dataset")

print("Path to dataset files:", path)