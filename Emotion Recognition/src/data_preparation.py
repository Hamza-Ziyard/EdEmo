import os

# prepare folder structure
main_folders =  ["test","train"]
sub_folders = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for main_folder in main_folders:
    os.makedirs(os.path.join('data',main_folder), exist_ok=True)
    for sub_folder in sub_folders:
        os.makedirs(os.path.join('data',main_folder,sub_folder), exist_ok=True)