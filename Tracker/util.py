import numpy as np
from shapely.geometry import Polygon

def get_bev_polygon(box_params):
    """
    Generate BEV(Bird's Eye View) 2D polygon from 3D box information.
    Using Shapely library makes it easy to calculate intersection area of rotated rectangles.
    
    Args:
        box_params (list): List in [cx, cy, cz, w, l, h, yaw] format
        
    Returns:
        shapely.geometry.Polygon: 2D polygon object in BEV plane
    """
    cx, cy, _, w, l, _, yaw = box_params
    # Generate basic rectangle corner coordinates using box l, w (center at 0,0)
    corners = np.array([
        [-w/2, -l/2], [-w/2, l/2], [w/2, l/2], [w/2, -l/2]
    ])
    # Generate and apply yaw rotation matrix
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    rotated_corners = corners @ rotation_matrix.T
    # Move to actual cx, cy center
    return Polygon(rotated_corners + np.array([cx, cy]))

def calculate_3d_iou(box1_params, box2_params):
    """
    Approximate calculation of IoU between two 3D bounding boxes (BEV IoU * Z-axis IoU).
    Since exact 3D IoU is very complex, it's efficient to calculate BEV and Z-axis separately.
    
    Args:
        box1_params (list): First box information [cx, cy, cz, w, l, h, yaw]
        box2_params (list): Second box information [cx, cy, cz, w, l, h, yaw]

    Returns:
        float: Calculated 3D IoU value
    """
    # 1. Calculate BEV (Bird's Eye View) IoU
    poly1 = get_bev_polygon(box1_params)
    poly2 = get_bev_polygon(box2_params)
    
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    bev_iou = intersection_area / union_area if union_area > 0 else 0

    # 2. Calculate Z-axis IoU
    cz1, h1 = box1_params[2], box1_params[5]
    cz2, h2 = box2_params[2], box2_params[5]
    z_min1, z_max1 = cz1 - h1/2, cz1 + h1/2
    z_min2, z_max2 = cz2 - h2/2, cz2 + h2/2
    
    z_intersection = max(0, min(z_max1, z_max2) - max(z_min1, z_min2))
    z_union = (z_max1 - z_min1) + (z_max2 - z_min2) - z_intersection
    z_iou = z_intersection / z_union if z_union > 0 else 0
        
    return bev_iou * z_iou