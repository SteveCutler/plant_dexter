from PIL import Image
import os

input_dir = "data/plant_images"
output_dir = "data/plant_images_webp"
os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(input_dir):
    # mirror the folder structure in the output
    rel_path = os.path.relpath(root, input_dir)
    target_dir = os.path.join(output_dir, rel_path)
    os.makedirs(target_dir, exist_ok=True)

    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path_in = os.path.join(root, fname)
        out_name = os.path.splitext(fname)[0] + ".webp"
        path_out = os.path.join(target_dir, out_name)

        try:
            img = Image.open(path_in).convert("RGB")
            img.thumbnail((512, 512))
            img.save(path_out, "WEBP", quality=70, method=6)
        except Exception as e:
            print(f"❌ Error processing {path_in}: {e}")

print("✅ All images compressed recursively to WebP.")
