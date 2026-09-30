from PIL import Image
import os

input_folder = "input_images"
output_folder = "resized_images"

new_size = (640, 640)  # width, height

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = Image.open(input_path)
        image = image.resize(new_size)

        image.save(output_path)

        print(f"Resized: {filename}")

print("Done.")