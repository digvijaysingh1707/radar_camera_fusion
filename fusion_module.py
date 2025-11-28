import numpy as np

class FusionEngine:
    def __init__(self):
        pass

    def fuse(self, camera_detections, radar_detections):
        """
        Simple fusion: Associate radar points to camera boxes.
        
        Args:
            camera_detections: List of [x1, y1, x2, y2, conf, cls]
            radar_detections: List of dicts {'range', 'azimuth', 'velocity', ...}
            
        Returns:
            fused_objects: List of dicts containing combined info.
        """
        fused_objects = []
        
        # If no radar data, just return camera data formatted
        if not radar_detections:
            for det in camera_detections:
                fused_objects.append({
                    'bbox': det[:4],
                    'class_id': int(det[5]),
                    'conf': det[4],
                    'radar_data': None
                })
            return fused_objects

        # Simple Association: Match Radar Azimuth to Camera X-coordinate
        # In a real system, we'd project radar to image plane or camera to bird's eye view.
        # Here we use the simplified Mock logic in reverse.
        
        used_radar_indices = set()
        
        for cam_det in camera_detections:
            box_center_x = (cam_det[0] + cam_det[2]) / 2
            # Estimated Azimuth from Camera
            cam_azimuth = ((box_center_x / 640.0) - 0.5) * 60.0
            
            best_match = None
            min_diff = 10.0 # Degree threshold
            best_idx = -1
            
            for i, rad_det in enumerate(radar_detections):
                if i in used_radar_indices:
                    continue
                
                diff = abs(rad_det['azimuth'] - cam_azimuth)
                if diff < min_diff:
                    min_diff = diff
                    best_match = rad_det
                    best_idx = i
            
            fused_obj = {
                'bbox': cam_det[:4],
                'class_id': int(cam_det[5]),
                'conf': cam_det[4],
                'radar_data': None
            }
            
            if best_match:
                fused_obj['radar_data'] = best_match
                used_radar_indices.add(best_idx)
                
            fused_objects.append(fused_obj)
            
        return fused_objects
