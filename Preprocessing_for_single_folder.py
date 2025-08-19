import os
import cv2
from PIL import Image


def select_crop_area(image_path):
    # Open the image using OpenCV
    img = cv2.imread(image_path)

    # Allow the user to select the region of interest (ROI)
    r = cv2.selectROI("Select Crop Area", img, fromCenter=False, showCrosshair=True)

    # Close the selection window
    cv2.destroyWindow("Select Crop Area")

    # Return the cropping rectangle as (x, y, width, height)
    return r


def crop_images(src_root, crop_box):
    # Crop and save images in the folder
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.jpg'):
                src_file = os.path.join(root, file)

                # Open the image using Pillow
                with Image.open(src_file) as img:
                    # Crop the image using the provided crop box
                    left, top, right, bottom = crop_box
                    cropped_img = img.crop((left, top, right, bottom))

                    # Save the cropped image (overwrite or save to new directory)
                    cropped_img.save(src_file)
                    print(f"Cropped and saved {src_file}")


if __name__ == "__main__":
    # Define the source root directory
    src_root = r"D:\PhD topic\Confucal Microscopy Project\New_dataset\data"

    # Select an image for manually selecting the crop area
    sample_image = os.path.join(src_root, os.listdir(src_root)[0])

    # Let the user select the cropping area
    x, y, w, h = select_crop_area(sample_image)

    # Define the crop box (left, upper, right, lower)
    crop_box = (x, y, x + w, y + h)

    # Crop all images using the selected crop box
    crop_images(src_root, crop_box)

