from itertools import combinations

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        max_area = -1
        point_set = set((x, y) for x, y in points)
        
        # Iterate through all combinations of 4 points
        for rect_points in combinations(points, 4):
            # Extract x and y coordinates
            x_coords = sorted(list(set(p[0] for p in rect_points)))
            y_coords = sorted(list(set(p[1] for p in rect_points)))
            
            # Check if it forms a rectangle with sides parallel to axes
            if len(x_coords) == 2 and len(y_coords) == 2:
                # Define the rectangle's corners
                rect_corners = [(x_coords[0], y_coords[0]),
                                (x_coords[0], y_coords[1]),
                                (x_coords[1], y_coords[0]),
                                (x_coords[1], y_coords[1])]
                
                # Check if all corners are in the point set
                if all(corner in point_set for corner in rect_corners):
                    # Calculate area
                    width = x_coords[1] - x_coords[0]
                    height = y_coords[1] - y_coords[0]
                    area = width * height
                    
                    # Check if no other points are inside or on the border
                    valid = True
                    for x in range(x_coords[0], x_coords[1] + 1):
                        for y in range(y_coords[0], y_coords[1] + 1):
                            if (x, y) not in rect_corners and (x, y) in point_set:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        if area > max_area:
                            max_area = area
        
        return max_area