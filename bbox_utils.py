import numpy as np
import cv2

def get_bounding_boxes(heatmap):
    """
    This function returns the bounding boxes of the objects in a binary heatmap.

    Parameters:
    heatmap (numpy.ndarray): 2D array representing the binary heatmap. The values
                             in the array should be either 0 or 1, or 0 to 255.

    Returns:
    list: A list of tuples representing the bounding boxes of the objects. Each
          tuple contains four values (x, y, w, h) representing the top-left 
          coordinate (x, y) and the width and height (w, h) of the bounding box.
    """
    try:
        # Check the value range of the heatmap
        if np.max(heatmap) <= 1:
            # Convert heatmap from [0,1] to [0,255]
            heatmap = (heatmap * 255).astype(np.uint8)
        elif np.max(heatmap) <= 255:
            # Ensure heatmap is of type uint8
            heatmap = heatmap.astype(np.uint8)
        else:
            raise ValueError("Invalid heatmap values. Expected values in range [0, 1] or [0, 255].")

        # Find contours in the binary image
        contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            # Compute the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        return bounding_boxes
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def main():
  print('Executing main...')



if __name__ == "__main__":
  main()
