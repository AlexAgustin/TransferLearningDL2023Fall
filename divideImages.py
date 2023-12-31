import os
import random
import shutil

# Path to the main directory containing the 64 subdirectories
main_directory = ''

# Names of the new directories
train_dir = ''
validation_dir = ''
test_dir = ''

# Percentage of images for each set
train_percentage = 70
validation_percentage = 15
test_percentage = 15

# Function to move random images to a new directory
def move_random_images(source_dir, dest_dir, num_images, exclude_set):
    filenames = [filename for filename in os.listdir(source_dir) if filename not in exclude_set]
    
    if len(filenames) < num_images:
        print(f"Not enough images in {source_dir}. Skipping this directory.")
        return
    
    selected_images = random.sample(filenames, num_images)
    exclude_set.update(selected_images)
    
    for image in selected_images:
        source_path = os.path.join(source_dir, image)
        dest_path = os.path.join(dest_dir, image)
        shutil.move(source_path, dest_path)

# Iterate over the 64 directories
for subdir in os.listdir(main_directory):
    # Exclude "train", "validation", and "test" directories
    if subdir not in ["train", "validation", "test"]:
        subdir_path = os.path.join(main_directory, subdir)
        
        # Initialize exclusion set to avoid duplicates
        exclude_set = set()
        
        # Calculate the number of images for each set
        num_images_train = int(len(os.listdir(subdir_path)) * (train_percentage / 100))
        num_images_validation = int(len(os.listdir(subdir_path)) * (validation_percentage / 100))
        num_images_test = int(len(os.listdir(subdir_path)) * (test_percentage / 100))
        
        print("Hooooooooooooo", subdir, num_images_train, num_images_validation, num_images_test)
        # Move images to the new directories
        move_random_images(subdir_path, train_dir, num_images_train, exclude_set)
        move_random_images(subdir_path, validation_dir, num_images_validation, exclude_set)
        move_random_images(subdir_path, test_dir, num_images_test, exclude_set)

print("Process completed.")
