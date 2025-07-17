import sys
import json
import cv2
import numpy as np
import os
import logging
from datetime import datetime
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {
            "detection": {
                "conf_threshold": 0.5,
                "nms_threshold": 0.4,
                "classes": ["helmet", "gloves", "mask", "safety_vest", "goggles"]
            },
            "tensorboard": {
                "log_dir": "logs/tensorboard"
            }
        }

def setup_tensorboard_logging(config):
    """Setup TensorBoard logging if available"""
    if not TENSORBOARD_AVAILABLE:
        logger.warning("TensorBoard not available. Install tensorboardX for logging.")
        return None
    
    try:
        log_dir = config.get("tensorboard", {}).get("log_dir", "logs/tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir)
    except Exception as e:
        logger.error(f"Failed to setup TensorBoard: {e}")
        return None

def load_yolo_model():
    """Load YOLO model configuration and weights with error handling"""
    try:
        weights_path = "models/yolov4.weights"
        config_path = "models/yolov4.cfg"
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"YOLO config file not found: {config_path}")
        
        logger.info("Loading YOLO model...")
        net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        logger.info("YOLO model loaded successfully")
        return net, output_layers
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

def detect_ppe(image_path, net, output_layers, config, writer=None):
    """Detect PPE in image with enhanced error handling and logging"""
    try:
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        height, width, channels = img.shape
        logger.info(f"Processing image: {image_path} ({width}x{height})")

        # Create blob and perform forward pass
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Load class names from config
        classes = config.get("detection", {}).get("classes", ["helmet", "gloves", "mask"])
        conf_threshold = config.get("detection", {}).get("conf_threshold", 0.5)
        nms_threshold = config.get("detection", {}).get("nms_threshold", 0.4)

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and class_id < len(classes):
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        results = []
        detected_items = {}
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                class_name = classes[class_ids[i]]
                confidence = confidences[i]
                
                # Determine compliance status based on detection
                status = "compliant" if confidence > 0.7 else "uncertain"
                
                result = {
                    "type": class_name,
                    "status": status,
                    "confidence": confidence,
                    "bbox": box,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                
                # Count detected items for logging
                detected_items[class_name] = detected_items.get(class_name, 0) + 1

        # Log metrics to TensorBoard if available
        if writer and TENSORBOARD_AVAILABLE:
            try:
                writer.add_scalar('Detection/Total_Objects', len(results), int(datetime.now().timestamp()))
                writer.add_scalar('Detection/Average_Confidence', 
                                np.mean([r['confidence'] for r in results]) if results else 0, 
                                int(datetime.now().timestamp()))
                for item_type, count in detected_items.items():
                    writer.add_scalar(f'Detection/{item_type}_Count', count, int(datetime.now().timestamp()))
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")

        # Calculate overall compliance
        total_required_ppe = len(classes)
        detected_ppe_types = len(set(r['type'] for r in results))
        compliance_rate = (detected_ppe_types / total_required_ppe) * 100 if total_required_ppe > 0 else 0

        logger.info(f"Detection complete: {len(results)} objects detected, {compliance_rate:.1f}% compliance")
        
        return {
            "success": True,
            "detections": results,
            "summary": {
                "total_objects": len(results),
                "compliance_rate": compliance_rate,
                "detected_ppe_types": list(detected_items.keys()),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"success": False, "error": "No image path provided"}))
            sys.exit(1)

        image_path = sys.argv[1]
        config = load_config()
        writer = setup_tensorboard_logging(config)
        
        net, output_layers = load_yolo_model()
        results = detect_ppe(image_path, net, output_layers, config, writer)
        
        if writer:
            writer.close()
            
        print(json.dumps(results))
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        print(json.dumps({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)
