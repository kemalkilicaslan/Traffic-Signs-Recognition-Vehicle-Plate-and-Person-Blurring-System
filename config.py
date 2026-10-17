# Traffic Signs Recognition, Vehicle Plate and Person Blurring System — Configuration

# Model weights file names
YOLO_MODEL_PATH          = "yolo11l.pt"                  # YOLOv11 model for person, car, motorcycle, bus, truck detection
TRAFFIC_SIGNS_MODEL_PATH = "Traffic-Signs-Recognition.pt"  # Model for traffic signs detection
VEHICLE_PLATE_MODEL_PATH = "Vehicle-Plates-in-Turkey.pt"   # Model for vehicle plate detection

# Video file names
INPUT_VIDEO  = "Traffic.mp4"  # input video file name
OUTPUT_VIDEO = "Traffic-Signs-Recognition-Vehicle-Plate-and-Person-Blurring.mp4"  # output video file name

# Detection confidence thresholds
TRAFFIC_SIGN_CONFIDENCE = 0.8   # minimum confidence to accept a traffic sign detection
DETECTION_CONFIDENCE    = 0.55  # minimum confidence for person / vehicle (YOLOv11) detection
PLATE_CONFIDENCE        = 0.55  # minimum confidence for vehicle plate detection

# Class name used to trigger person blurring
PERSON_CLASS = "person"

# Vehicle class names whose plates are detected and blurred
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Gaussian blur settings (kernel width/height must be odd) for masking regions
PERSON_BLUR_KERNEL = (99, 99)   # blur kernel applied to person regions
PERSON_BLUR_SIGMA  = 30         # blur strength (sigma) for person regions
PLATE_BLUR_KERNEL  = (255, 255) # blur kernel applied to vehicle plate regions
PLATE_BLUR_SIGMA   = 30         # blur strength (sigma) for vehicle plate regions

# Traffic sign label drawing (OpenCV putText / rectangle)
LABEL_FONT_SCALE     = 1                          # font scale for the sign label text
LABEL_FONT_THICKNESS = 3                          # thickness of the sign label text
LABEL_TEXT_COLOR     = (0, 0, 0)                  # BGR color of the label text (black)
LABEL_BG_COLOR       = (255, 255, 255)            # BGR background color behind the label (white)
BOX_COLOR            = (255, 255, 255)            # BGR color of the sign bounding box (white)
BOX_THICKNESS        = 5                          # thickness of the sign bounding box

# Set to True to show the live annotated window (cv2.imshow); disable for headless runs
SHOW_WINDOW = True
