import torch
import cv2
import warnings

# Suppress specific FutureWarning for context manager
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*autocast.*is deprecated.*")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load video
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)
# Desired output size for display
frame_width = 1080 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = 720 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_size = (frame_width, frame_height)
# Define video writer to save output
output_path = 'output_video.avi'
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

# Mapping COCO classes to custom vehicle types
vehicle_class_mapping = {
    'bicycle': 'LV',
    'car': 'LV',
    'motorcycle': '2_wheeler',
    'bus': 'HV',
    'truck': 'HV'
}

# Box colors for each custom vehicle type
box_colors = {
    'LV': (255, 0, 0),        # Blue for Light Vehicles
    '2_wheeler': (0, 255, 0), # Green for 2 Wheelers
    'HV': (0, 0, 255)         # Red for Heavy Vehicles
}



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, output_size)
    # Convert BGR image to RGB and run detection
    results = model(frame[..., ::-1])

    # Get predictions
    predictions = results.xyxy[0]

    # Initialize a dictionary to count vehicle types for the current frame
    frame_vehicle_count = {'LV': 0, '2_wheeler': 0, 'HV': 0}

    # Count detected vehicles in the current frame
    for *box, conf, cls in predictions:
        class_name = model.names[int(cls)]
        if class_name in vehicle_class_mapping:
            category = vehicle_class_mapping[class_name]
            frame_vehicle_count[category] += 1

            # Draw bounding box with the appropriate color
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_colors[category], 2)
            label = f"{category}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[category], 2)

    # Calculate the total number of vehicles detected in the current frame
    total_vehicles = sum(frame_vehicle_count.values())

    # Display the count of vehicles on the top right corner
    offset_y = 30
    for i, (vehicle_type, count) in enumerate(frame_vehicle_count.items()):
        text = f"{vehicle_type}: {count}"
        cv2.putText(frame, text, (frame_width - 250, 30 + i * offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    total_text = f"Total: {total_vehicles}"
    cv2.putText(frame, total_text, (frame_width - 250, 30 + len(frame_vehicle_count) * offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (Optional)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
