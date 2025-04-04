from ultralytics import YOLO
import cv2

# Load the trained model
model_path = "yolo_saved.pt"  # Path to the best model weights
model = YOLO(model_path)

# Path to the test image
image_path = "dataset/images/train/000117.jpg"

# Run inference on the image
results = model(image_path)

# Debugging: Print results
print("Results for image:", image_path)
for result in results:
    print("Detected objects:")
    for box in result.boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Bounding Box: {box.xyxy}")

# Save the results
output_image_path = "output_117.jpg"
results[0].save(output_image_path)
print(f"Results saved to: {output_image_path}")

# Display the results (optional)
cv2.imshow("Output", cv2.imread(output_image_path))
cv2.waitKey(0)
cv2.destroyAllWindows()