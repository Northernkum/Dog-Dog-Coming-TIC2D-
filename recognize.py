import cv2
import numpy as np
import mediapipe as mp
import math
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


DISPLAY_WIDTH = 640

ChannelFactoryInitialize(0)
client = VideoClient()
client.SetTimeout(3.0)
client.Init()

class OKGestureRecognizer:
    def __init__(self):
      
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
       
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
       
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.last_ok_detected = False
        self.ok_count = 0
        self.confirm_threshold = 3 
        
    def calculate_distance(self, point1, point2):
       
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    
    def is_ok_gesture_simple(self, landmarks, image_shape):
      
        if not landmarks or len(landmarks.landmark) < 21:
            return False
        
        h, w = image_shape[:2]
        
      
        points = {}
        for i, landmark in enumerate(landmarks.landmark):
            points[i] = (int(landmark.x * w), int(landmark.y * h))
        
        palm_size = self.calculate_distance(points[self.WRIST], points[9])  
        
        
        thumb_index_dist = self.calculate_distance(points[self.THUMB_TIP], points[self.INDEX_TIP])
        
        
        if thumb_index_dist < palm_size * 0.2:
           
            thumb_middle_dist = self.calculate_distance(points[self.THUMB_TIP], points[self.MIDDLE_TIP])
            thumb_ring_dist = self.calculate_distance(points[self.THUMB_TIP], points[self.RING_TIP])
            thumb_pinky_dist = self.calculate_distance(points[self.THUMB_TIP], points[self.PINKY_TIP])
            
            if (thumb_middle_dist > thumb_index_dist * 2 and 
                thumb_ring_dist > thumb_index_dist * 2 and 
                thumb_pinky_dist > thumb_index_dist * 2):
                return True
        
        return False
    
    def is_ok_gesture_advanced(self, landmarks, image_shape):
        
        if not landmarks or len(landmarks.landmark) < 21:
            return False
        
        h, w = image_shape[:2]
        
        
        points = {}
        for i, landmark in enumerate(landmarks.landmark):
            points[i] = (int(landmark.x * w), int(landmark.y * h))
        
        palm_size = self.calculate_distance(points[self.WRIST], points[9])
        
        thumb_index_dist = self.calculate_distance(points[self.THUMB_TIP], points[self.INDEX_TIP])
        
        if thumb_index_dist < palm_size * 0.25:  
           
            thumb_direction = self.calculate_direction(points[2], points[self.THUMB_TIP])  
            index_direction = self.calculate_direction(points[5], points[self.INDEX_TIP]) 
            
            angle_diff = self.angle_difference(thumb_direction, index_direction)
            
            if abs(angle_diff - 180) < 60:  
                return True
        
        return False
    
    def calculate_direction(self, point1, point2):
      
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return 0, 0
        return dx/length, dy/length
    
    def angle_difference(self, dir1, dir2):
     
        dot_product = dir1[0]*dir2[0] + dir1[1]*dir2[1]
    
        dot_product = max(-1, min(1, dot_product))
        angle_rad = math.acos(dot_product)
        return math.degrees(angle_rad)
    
    def process_frame(self, frame):
       
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        ok_detected = False
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
      
            simple_ok = self.is_ok_gesture_simple(results.multi_hand_landmarks[0], frame.shape)
            advanced_ok = self.is_ok_gesture_advanced(results.multi_hand_landmarks[0], frame.shape)
            
            if simple_ok and advanced_ok:
                ok_detected = True
                confidence = 2
            elif simple_ok or advanced_ok:
                ok_detected = True
                confidence = 1
            
            if ok_detected:
             
                h, w = frame.shape[:2]
                color = (0, 255, 0) if confidence == 2 else (0, 0, 255) 
                cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 3)
                text = "OK Gesture Detected"
                if confidence == 2:
                    text += " (High Confidence)"
                else:
                    text += " (Low Confidence)"
                cv2.putText(frame, text, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if ok_detected:
            self.ok_count += 1
            if self.ok_count >= self.confirm_threshold and not self.last_ok_detected:
                print("detect ok")
                self.last_ok_detected = True
        else:
            self.ok_count = 0
            self.last_ok_detected = False
        
        return frame

gesture_recognizer = OKGestureRecognizer()

print("run seccess")
print("do ok")
print("press esc to exit")

while True:
 
    code, data = client.GetImageSample()
    if code != 0:
        print("fail", code)
        continue

    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if frame is None:
        print("fail")
        continue

    original_h, original_w = frame.shape[:2]
    scale_factor = DISPLAY_WIDTH / original_w
    display_h = int(original_h * scale_factor)
    display_w = DISPLAY_WIDTH

    processed_frame = gesture_recognizer.process_frame(frame)
    
    frame_to_show = cv2.resize(processed_frame, (display_w, display_h))
    cv2.imshow("Improved OK Gesture Recognition", frame_to_show)

    if cv2.waitKey(1) & 0xFF == 27:
        break

client.Release()
cv2.destroyAllWindows()
print("exit ok")
