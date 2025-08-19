import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import glob


class CropApp:
    def __init__(self, root, img_path, output_folder):
        self.root = root
        self.img_path = img_path
        self.output_folder = output_folder
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.curX = None
        self.curY = None
        self.image = None
        self.canvas = None
        self.crop_coords = None

        self.init_ui()

    def init_ui(self):
        # Load image
        self.image = Image.open(self.img_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        # Add image to canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Add save button
        btn_save = tk.Button(self.root, text="Crop and Save", command=self.crop_and_save)
        btn_save.pack()

    def on_button_press(self, event):
        # Save the initial coordinates
        self.start_x = event.x
        self.start_y = event.y

        # Create a rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                     outline='red')

    def on_mouse_drag(self, event):
        self.curX, self.curY = (event.x, event.y)

        # Expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)

    def on_button_release(self, event):
        # Final coordinates
        self.curX, self.curY = (event.x, event.y)
        self.crop_coords = (self.start_x, self.start_y, self.curX, self.curY)
        print(f"Selected coordinates: {self.crop_coords}")

    def crop_and_save(self):
        if not self.crop_coords:
            print("No crop coordinates selected.")
            return

        for img_path in glob.glob(os.path.join(os.path.dirname(self.img_path), "*.jpg")):
            img = Image.open(img_path)
            cropped_img = img.crop(self.crop_coords)

            # Save cropped image
            base_name = os.path.basename(img_path)
            output_path = os.path.join(self.output_folder, base_name)
            cropped_img.save(output_path)
            print(f"Cropped image saved as: {output_path}")

        print("Cropping and saving completed.")
        self.root.quit()


def main():
    # Define the source directory containing images and the destination directory for cropped images
    src_dir = r"D:\PhD topic\Confucal dataset\CRS_Processed"
    dest_dir = r"D:\PhD topic\Confucal dataset\CRS_Data_Aug_original"

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Find the first image in the source directory
    first_image_path = glob.glob(os.path.join(src_dir, "*.jpg"))[0]

    root = tk.Tk()
    app = CropApp(root, first_image_path, dest_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
