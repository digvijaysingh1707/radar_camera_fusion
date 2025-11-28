import numpy as np
import time

class MockRadarDetector:
    def __init__(self):
        self.last_update = time.time()

    def detect(self, camera_objects=None):
        """
        Simulates radar detections.
        If camera_objects is provided (list of boxes), it generates radar points 
        that roughly match the camera objects to simulate 'good' fusion.
        Otherwise, it returns random noise.
        
        Returns:
            detections (list): List of [range, azimuth, velocity]
        """
        detections = []
        
        # Simulate targets based on camera objects for demonstration
        if camera_objects:
            for obj in camera_objects:
                # obj is typically [x1, y1, x2, y2, conf, cls]
                # We estimate range based on box height (simple heuristic)
                box_h = obj[3] - obj[1]
                box_center_x = (obj[0] + obj[2]) / 2
                
                # Heuristic: smaller box = further away
                # range ~ constant / height
                # This is just for visual correlation in the demo
                sim_range = 500.0 / (box_h + 1e-6) 
                
                # Azimuth based on x position (0 is center)
                # Normalize x to -1 to 1, then scale to degrees
                # Assuming 640px width
                sim_azimuth = ((box_center_x / 640.0) - 0.5) * 60.0 # +/- 30 degrees
                
                sim_velocity = np.random.normal(0, 2.0) # Random velocity
                
                # Add some noise
                sim_range += np.random.normal(0, 0.5)
                sim_azimuth += np.random.normal(0, 1.0)
                
                detections.append({
                    'range': sim_range,
                    'azimuth': sim_azimuth,
                    'velocity': sim_velocity,
                    'id': int(obj[5]) # Class ID
                })
        
        # Add some clutter/noise
        if np.random.random() > 0.7:
             detections.append({
                'range': np.random.uniform(5, 50),
                'azimuth': np.random.uniform(-45, 45),
                'velocity': np.random.normal(0, 5),
                'id': -1 # Noise
            })
            
        return detections
