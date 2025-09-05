from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from util import calculate_3d_iou

import numpy as np


class Track:
    """Class that manages information for a single tracked object"""
    def __init__(self, detection, feature, track_id):
        """
        Initialize Track object
        
        Args:
            detection (list): [cx, cy, cz, w, l, h, yaw, score, cls_id]
            feature (np.array): Feature vector of the object
            track_id (int): Unique tracking ID
        """
        self.id = track_id
        self.box_params = detection[:7]  # [cx, cy, cz, w, l, h, yaw]
        self.score = detection[7]
        self.cls_id = int(detection[8])
        self.row_idx = int(detection[9])
        self.feature = feature
        
        # Kalman filter setup (estimate cx, cy position and velocity)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables, 2 measurement variables
        # State vector [x, y, vx, vy]. Initial velocity set to 0
        self.kf.x = np.array([self.box_params[0], self.box_params[1], 0, 0])
        
        # State transition matrix (F): defines how previous state affects next state
        # (x_t = x_{t-1} + vx_{t-1}*dt)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                              [0, 0, 1, 0], [0, 0, 0, 1]])
        
        # Measurement matrix (H): defines which state variables can be measured (only x, y position here)
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Initialize covariance matrices
        self.kf.P = np.diag([100., 100., 1000., 1000.])
        self.kf.R = np.diag([1.0, 1.0]) 
        self.kf.Q = np.diag([0.1, 0.1, 0.1, 0.1])  # Process noise (model uncertainty)
        
        self.time_since_update = 0  # Number of frames since last update
        self.hits = 1               # Number of consecutive successful matches
        self.history = []           # List to store past states (used if needed)

    def predict(self):
        """Kalman filter prediction step. Predicts the object state for the current frame."""
        self.kf.predict()
        self.time_since_update += 1
        # Reflect predicted position [x, y] to current box information
        self.box_params[0] = self.kf.x[0]
        self.box_params[1] = self.kf.x[1]

    def update(self, detection, feature):
        """
        Kalman filter update step. Updates the state with matched detection information.
        """
        self.box_params = detection[:7]
        self.score = detection[7]
        self.row_idx = int(detection[9])
        self.cls_id = int(detection[8])
        if feature is not None:
            if self.feature is None:
                self.feature = feature
            else:        
                self.feature = self.feature*0.9 + feature*0.1
                self.feature = self.feature / np.linalg.norm(self.feature)  # Normalize
        
        # Update Kalman filter with matched detection's [cx, cy]
        measurement = np.array([self.box_params[0], self.box_params[1]])
        self.kf.update(measurement)
        
        # Update state
        self.time_since_update = 0
        self.hits += 1

class DeepSort3D:
    """
    Main tracker class applying SORT algorithm to 3D.
    Manages multiple Track objects and performs data association.
    """
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.1):
        """
        Initialize tracker
        
        Args:
            max_age (int): How many frames a track can be invisible before deletion
            min_hits (int): How many consecutive matches needed to consider a track stable
            iou_threshold (float): Minimum IoU value to consider two boxes as the same object
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0

    def update(self, detections_raw,features=None):
        """
        Main function called every frame to perform tracking
        
        Args:
            detections_raw (np.array): All detection information for current frame
                                       (N, 9) size, [cx..cy..cz..w..l..h..yaw..score..cls_id]
                                       
        Returns:
            np.array: Tracking object information (M, 10) size
                      [cx..cy..cz..w..l..h..yaw..score..track_id..cls_id]
        """
        # 1. Predict next state for existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Association between predicted tracks and new detections
        # Generate cost matrix (cost = 1 - IoU)
        num_tracks = len(self.tracks)
        num_dets = len(detections_raw)
        cost_matrix = np.ones((num_tracks, num_dets))

        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections_raw):
                iou = calculate_3d_iou(track.box_params, det[:7])
                if track.feature is None or features is None:
                    feat_distance = 0.0
                elif track.cls_id == 0 or track.cls_id == 4 or track.cls_id == 5:
                    feat_distance = np.linalg.norm(track.feature - features[d]) if features is not None else 0.0
                else:
                    feat_distance = 0.0
                if iou > self.iou_threshold and feat_distance < 0.5:
                    cost_matrix[t, d] = (1 - iou)# + feat_distance * 0.1 don't do this  # Cost matrix combining IoU and feature vector distance
        
        # Find optimal matching using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_indices = []
        # Only accept matches that pass IoU threshold
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - self.iou_threshold):
                matched_indices.append((r, c))

        unmatched_tracks = set(range(num_tracks))
        unmatched_dets = set(range(num_dets))
        for r, c in matched_indices:
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        # 3. Update tracks
        # 3.1. Update matched tracks with detection information
        for r, c in matched_indices:
            if features is None:
                self.tracks[r].update(detections_raw[c], None)
            else:
                self.tracks[r].update(detections_raw[c],features[c])

        # 3.2. Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            if features is None:
                new_track = Track(detections_raw[d_idx], None, self.next_id)
            else:
                new_track = Track(detections_raw[d_idx], features[d_idx], self.next_id)
            self.tracks.append(new_track)
            self.next_id += 1

        # 4. Clean up tracks and return results
        active_tracks = []
        final_results = []
        for track in self.tracks:
            # Delete tracks that haven't been updated for more than max_age frames
            if track.time_since_update <= self.max_age:
                active_tracks.append(track)
                # Only output stable tracks that satisfy min_hits condition
                if track.hits >= self.min_hits:
                    # [cx, cy, cz, w, l, h, yaw, score, track_id, cls_id]
                    res = np.concatenate((track.box_params, np.array([track.score, track.id, track.cls_id,track.row_idx])))
                    final_results.append(res)
        
        self.tracks = active_tracks
        return np.array(final_results)