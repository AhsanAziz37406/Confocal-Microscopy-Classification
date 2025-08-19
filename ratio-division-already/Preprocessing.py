import os
import shutil


def collect_images(src_root, dest_root):
    # Define destination folders
    n_dest_folder = os.path.join(dest_root, "N_P")
    t_dest_folder = os.path.join(dest_root, "T_P")

    # Create destination folders if they don't exist
    os.makedirs(n_dest_folder, exist_ok=True)
    os.makedirs(t_dest_folder, exist_ok=True)

    # Walk through the source directory
    for root, dirs, files in os.walk(src_root):
        # Check if the current directory is named "N" or "T"
        if os.path.basename(root) in ['N', 'T']:
            for file in files:
                if file.endswith('.jpg'):
                    src_file = os.path.join(root, file)
                    if os.path.basename(root) == 'N':
                        dest_file = os.path.join(n_dest_folder, file)
                    else:
                        dest_file = os.path.join(t_dest_folder, file)
                    shutil.copy2(src_file, dest_file)
                    print(f"Copied {src_file} to {dest_file}")


if __name__ == "__main__":
    # Define the source and destination root directories
    src_root = r"D:\PhD topic\Confucal Microscopy Project\New_dataset\NEW_CRS AI image_CLE"
    dest_root = r"D:\PhD topic\Confucal Microscopy Project\New_dataset\data"

    # Collect and copy images
    collect_images(src_root, dest_root)

