import sys
import json
import cv2
import numpy as np
import os
import logging
from datetime import datetime
import base64

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.authorized_faces_dir = "authorized_faces"
        self.model_path = "models/face_recognition_model.yml"
        self.labels = {}
        self.label_counter = 0
        
        # Create directories if they don't exist
        os.makedirs(self.authorized_faces_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
    def load_authorized_faces(self):
        """Load and train the face recognition model with authorized faces"""
        try:
            faces = []
            labels = []
            
            # Load existing model if available
            if os.path.exists(self.model_path):
                self.face_recognizer.read(self.model_path)
                logger.info("Loaded existing face recognition model")
            
            # Scan authorized faces directory
            for person_name in os.listdir(self.authorized_faces_dir):
                person_dir = os.path.join(self.authorized_faces_dir, person_name)
                if os.path.isdir(person_dir):
                    self.labels[self.label_counter] = person_name
                    
                    for image_name in os.listdir(person_dir):
                        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(person_dir, image_name)
                            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                            
                            if image is not None:
                                face_locations = self.face_cascade.detectMultiScale(
                                    image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                                )
                                
                                for (x, y, w, h) in face_locations:
                                    face_roi = image[y:y+h, x:x+w]
                                    faces.append(face_roi)
                                    labels.append(self.label_counter)
                    
                    self.label_counter += 1
            
            # Train the model if we have faces
            if faces:
                self.face_recognizer.train(faces, np.array(labels))
                self.face_recognizer.save(self.model_path)
                logger.info(f"Trained face recognition model with {len(faces)} face samples")
                return True
            else:
                logger.warning("No authorized faces found for training")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load authorized faces: {e}")
            return False
    
    def verify_face_from_image(self, image_path):
        """Verify face from an uploaded image"""
        try:
            if not os.path.exists(self.model_path):
                return {
                    "success": False,
                    "error": "No trained model found. Please add authorized faces first.",
                    "confidence": 0,
                    "person": None
                }
            
            # Load and process the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Detect faces in the image
            faces = self.face_cascade.detectMultiScale(
                image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return {
                    "success": False,
                    "error": "No face detected in the image",
                    "confidence": 0,
                    "person": None
                }
            
            # Use the largest face for recognition
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            face_roi = image[y:y+h, x:x+w]
            
            # Predict using the trained model
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Lower confidence means better match in LBPH
            threshold = 100  # Adjust based on your needs
            
            if confidence < threshold:
                person_name = self.labels.get(label, "Unknown")
                success = True
                logger.info(f"Face recognized as {person_name} with confidence {confidence}")
            else:
                person_name = None
                success = False
                logger.info(f"Face not recognized. Confidence: {confidence}")
            
            return {
                "success": success,
                "confidence": float(confidence),
                "person": person_name,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Face verification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0,
                "person": None
            }
    
    def add_authorized_face(self, person_name, image_path):
        """Add a new authorized face to the system"""
        try:
            person_dir = os.path.join(self.authorized_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Convert to grayscale and detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                raise ValueError("No face detected in the provided image")
            
            # Save the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            face_roi = image[y:y+h, x:x+w]
            
            # Save the face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"{person_name}_{timestamp}.jpg"
            face_path = os.path.join(person_dir, face_filename)
            cv2.imwrite(face_path, face_roi)
            
            logger.info(f"Added authorized face for {person_name}")
            
            # Retrain the model
            self.load_authorized_faces()
            
            return {
                "success": True,
                "message": f"Authorized face added for {person_name}",
                "face_path": face_path
            }
            
        except Exception as e:
            logger.error(f"Failed to add authorized face: {e}")
            return {
                "success": False,
                "error": str(e)
            }

def main():
    """Main function for command line usage"""
    try:
        if len(sys.argv) < 2:
            print(json.dumps({
                "success": False,
                "error": "Usage: python face_recognition.py <command> [args]"
            }))
            sys.exit(1)
        
        command = sys.argv[1]
        face_system = FaceRecognitionSystem()
        
        if command == "verify":
            if len(sys.argv) < 3:
                print(json.dumps({
                    "success": False,
                    "error": "Usage: python face_recognition.py verify <image_path>"
                }))
                sys.exit(1)
            
            # Load authorized faces
            if not face_system.load_authorized_faces():
                print(json.dumps({
                    "success": False,
                    "error": "No authorized faces found"
                }))
                sys.exit(1)
            
            image_path = sys.argv[2]
            result = face_system.verify_face_from_image(image_path)
            print(json.dumps(result))
            
        elif command == "add_face":
            if len(sys.argv) < 4:
                print(json.dumps({
                    "success": False,
                    "error": "Usage: python face_recognition.py add_face <person_name> <image_path>"
                }))
                sys.exit(1)
            
            person_name = sys.argv[2]
            image_path = sys.argv[3]
            result = face_system.add_authorized_face(person_name, image_path)
            print(json.dumps(result))
            
        elif command == "train":
            result = face_system.load_authorized_faces()
            print(json.dumps({
                "success": result,
                "message": "Model training completed" if result else "Training failed"
            }))
            
        else:
            print(json.dumps({
                "success": False,
                "error": f"Unknown command: {command}"
            }))
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        print(json.dumps({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
