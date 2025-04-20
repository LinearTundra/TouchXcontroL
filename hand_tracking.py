import cv2
import mediapipe as mp
import numpy as np

class HandPos:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize OpenCV camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
            
    def get_hand_positions(self):
        """
        Returns the positions of hand landmarks for all detected hands.
        Returns a list of dictionaries, where each dictionary contains:
        - 'landmarks': List of (x, y, z) coordinates for each hand landmark
        - 'handedness': 'Left' or 'Right'
        """
        success, image = self.cap.read()
        if not success:
            return []
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        hand_positions = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (left or right hand)
                handedness = results.multi_handedness[hand_idx].classification[0].label
                
                # Extract landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))
                
                hand_positions.append({
                    'landmarks': landmarks,
                    'handedness': handedness
                })
                
        return hand_positions, results, image
    
    def display_feed(self, window_name="Hand Tracking"):
        """
        Displays the camera feed with hand landmarks overlaid.
        Press 'q' to quit the display.
        """
        while True:
            # Get hand positions and the processed image
            hand_positions, results, image = self.get_hand_positions()
            
            # Draw hand landmarks on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display the image
            cv2.imshow(window_name, image)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def release(self):
        """Release the camera and MediaPipe resources"""
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
        
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release() 

if __name__ == "__main__" :
    try:
        hand_tracker = HandPos()
        print("Starting hand tracking. Press 'q' to quit.")
        hand_tracker.display_feed()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        hand_tracker.release()