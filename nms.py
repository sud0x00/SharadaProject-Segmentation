import numpy as np

def nms(bounding_boxes, threshold):
    if len(bounding_boxes) == 0:
        return []

    boxes = np.array(bounding_boxes)
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    selected_boxes = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = np.argsort(end_y)  # Sort by the bottom-right y-coordinate of the bounding boxes

    while order.size > 0:
        index = order[-1]  # Index of the bounding box with the highest bottom-right y-coordinate

        selected_boxes.append(bounding_boxes[index])

        # Compute coordinates of intersection
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection and union
        intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        union_area = areas[index] + areas[order[:-1]] - intersection_area

        # Compute intersection over union (IoU)
        iou = intersection_area / union_area

        # Keep bounding boxes where IoU is less than the threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds]

    return selected_boxes
