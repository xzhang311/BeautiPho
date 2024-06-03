import os, json
from PIL import Image, ImageDraw

img_file = "images/demo/demo1.jpg"
img_name = img_file.split('/')[-1].split('.')[0]
in_img = Image.open(img_file)
in_img = in_img.resize((512, 512))


with open('demo1.json', 'r') as f:
    data = json.load(f)

# Define bounding boxes and optional labels
# Bounding boxes are defined as tuples: (x_min, y_min, x_max, y_max)
bboxes = data['bboxes'] # Example bounding boxes
labels = data['labels'] # Optional labels for each bbox

# Create an ImageDraw object
draw = ImageDraw.Draw(in_img)

# Optional: Choose a color for the bounding box
bbox_color = 'red'  # Color of the bounding box

# Draw each bounding box
for bbox, label in zip(bboxes, labels):
    # Draw rectangle (bbox)
    draw.rectangle(bbox, outline=bbox_color, width=2)

    # Optionally draw label text near the bounding box
    label_position = (bbox[0], bbox[1] - 10)  # Position text slightly above the top-left corner
    draw.text(label_position, label, fill=bbox_color)

# Display the image
in_img.show()

# Optionally save the modified image
in_img.convert('RGB').save('annotated_image_ram++.jpg')