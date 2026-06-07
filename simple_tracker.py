import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

class SimpleTracker:
    """
    A self-contained Hungarian algorithm-based bounding box tracker.
    
    Tracks are represented as:
    {
        track_id: {
            "class": str,
            "centroids": deque(maxlen=30) of (x,y),
            "bbox": (x1,y1,x2,y2),
            "speed": float,
            "direction": (dx,dy),
            "age": int,
            "missed": int
        }
    }
    """
    def __init__(self, min_iou=0.1, max_missed=10):
        self.tracks = {}
        self.next_id = 0
        self.min_iou = min_iou
        self.max_missed = max_missed

    def _compute_iou(self, box1, box2):
        """Computes intersection-over-union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        intersection_area = max(0.0, x2_i - x1_i) * max(0.0, y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    def update(self, detections):
        """
        Updates the tracker with new detections.
        
        Args:
            detections (list of dicts): list of {"bbox": [x1, y1, x2, y2], "class_name": str}
            
        Returns:
            list of dicts: Active track summaries.
        """
        track_ids = list(self.tracks.keys())
        N = len(track_ids)
        M = len(detections)
        
        # 1. Predict new centroids and shifted bboxes using linear extrapolation
        predictions = {}
        for tid in track_ids:
            track = self.tracks[tid]
            centroids = track["centroids"]
            
            if len(centroids) >= 2:
                # Linear extrapolation from last two centroids
                c_old = centroids[-2]
                c_new = centroids[-1]
                dx = c_new[0] - c_old[0]
                dy = c_new[1] - c_old[1]
                pred_cx = c_new[0] + dx
                pred_cy = c_new[1] + dy
            else:
                # Fallback to current centroid if not enough history
                pred_cx = centroids[-1][0]
                pred_cy = centroids[-1][1]
            
            # Shift the bounding box according to predicted centroid change
            x1, y1, x2, y2 = track["bbox"]
            curr_cx = (x1 + x2) / 2.0
            curr_cy = (y1 + y2) / 2.0
            shift_x = pred_cx - curr_cx
            shift_y = pred_cy - curr_cy
            
            pred_bbox = (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)
            predictions[tid] = {
                "centroid": (pred_cx, pred_cy),
                "bbox": pred_bbox
            }

        matched_tracks = set()
        matched_detections = set()

        # 2. Match tracks and detections using IoU and Hungarian Algorithm
        if N > 0 and M > 0:
            cost_matrix = np.ones((N, M), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                track = self.tracks[tid]
                pred_bbox = predictions[tid]["bbox"]
                for j, det in enumerate(detections):
                    if track["class"] == det["class_name"]:
                        iou = self._compute_iou(pred_bbox, det["bbox"])
                        cost_matrix[i, j] = 1.0 - iou
                    else:
                        cost_matrix[i, j] = 1.0  # Max cost if classes don't match
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                iou = 1.0 - cost_matrix[r, c]
                if iou >= self.min_iou:
                    tid = track_ids[r]
                    det = detections[c]
                    track = self.tracks[tid]
                    
                    x1, y1, x2, y2 = det["bbox"]
                    new_centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    
                    track["centroids"].append(new_centroid)
                    track["bbox"] = (x1, y1, x2, y2)
                    track["missed"] = 0
                    track["age"] += 1
                    
                    # Compute speed and direction from last two centroids
                    if len(track["centroids"]) >= 2:
                        c_old = track["centroids"][-2]
                        c_new = track["centroids"][-1]
                        dx = c_new[0] - c_old[0]
                        dy = c_new[1] - c_old[1]
                        track["speed"] = float(np.hypot(dx, dy))
                        track["direction"] = (dx, dy)
                    else:
                        track["speed"] = 0.0
                        track["direction"] = (0.0, 0.0)
                    
                    matched_tracks.add(tid)
                    matched_detections.add(c)

        # 3. Handle unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track["missed"] += 1
                track["age"] += 1

        # 4. Remove tracks that exceed max missed frames limit
        to_remove = [tid for tid, track in self.tracks.items() if track["missed"] > self.max_missed]
        for tid in to_remove:
            del self.tracks[tid]

        # 5. Handle unmatched detections (create new tracks)
        for c, det in enumerate(detections):
            if c not in matched_detections:
                self.next_id += 1
                new_tid = self.next_id
                
                x1, y1, x2, y2 = det["bbox"]
                new_centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                
                new_centroids_deque = deque(maxlen=30)
                new_centroids_deque.append(new_centroid)
                
                self.tracks[new_tid] = {
                    "class": det["class_name"],
                    "centroids": new_centroids_deque,
                    "bbox": (x1, y1, x2, y2),
                    "speed": 0.0,
                    "direction": (0.0, 0.0),
                    "age": 1,
                    "missed": 0
                }

        # 6. Format and return active track dicts
        active_list = []
        for tid, track in self.tracks.items():
            active_list.append({
                "track_id": tid,
                "class": track["class"],
                "centroid": track["centroids"][-1],
                "bbox": track["bbox"],
                "speed": track["speed"],
                "direction": track["direction"]
            })
        return active_list

if __name__ == '__main__':
    print("Running synthetic scenario to test SimpleTracker...")
    tracker = SimpleTracker(min_iou=0.1, max_missed=5)
    
    # Simulate a car moving along a diagonal path
    # Frame 0: Car appears
    # Frame 1: Car moves
    # Frame 2: Car moves
    # Frame 3: Car occluded (no detection)
    # Frame 4: Car reappears
    # Frames 5-11: Car disappears completely (should be deleted after 5 missed frames)
    
    frames_detections = [
        # Frame 0
        [{"bbox": [10.0, 10.0, 30.0, 30.0], "class_name": "car"}],
        # Frame 1
        [{"bbox": [12.0, 12.0, 32.0, 32.0], "class_name": "car"}],
        # Frame 2
        [{"bbox": [14.0, 14.0, 34.0, 34.0], "class_name": "car"}],
        # Frame 3 (occlusion/missed detection)
        [],
        # Frame 4 (reappears)
        [{"bbox": [18.0, 18.0, 38.0, 38.0], "class_name": "car"}],
        # Frame 5 (disappears)
        [],
        # Frame 6
        [],
        # Frame 7
        [],
        # Frame 8
        [],
        # Frame 9
        [],
        # Frame 10 (should be removed by now since max_missed=5)
        []
    ]
    
    for idx, detections in enumerate(frames_detections):
        active_tracks = tracker.update(detections)
        print(f"\n--- Frame {idx} ---")
        print(f"Detections input: {detections}")
        print(f"Active tracks: {active_tracks}")
        
        # Verification assertions
        if idx == 0:
            assert len(active_tracks) == 1
            assert active_tracks[0]["track_id"] == 1
            assert active_tracks[0]["speed"] == 0.0
        elif idx == 1:
            assert len(active_tracks) == 1
            # Speed should be sqrt(2^2 + 2^2) = 2.8284
            assert abs(active_tracks[0]["speed"] - 2.8284) < 1e-4
        elif idx == 4:
            # Reassociated track should keep track_id = 1
            assert len(active_tracks) == 1
            assert active_tracks[0]["track_id"] == 1
        elif idx >= 10:
            # Track should be cleaned up
            assert len(active_tracks) == 0

    print("\nAll synthetic tests passed successfully!")
