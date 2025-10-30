class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        
        # Convert points to a set for O(1) lookup
        points_set = set(tuple(point) for point in points)
        
        max_area = -1
        
        for i in range(len(points)):
            x1, y1 = points[i]
            for j in range(len(points)):
                if i == j:
                    continue
                    
                x2, y2 = points[j]
                
                # Points should not share x or y coordinates to form diagonal
                if x1 == x2 or y1 == y2:
                    continue
                
                # Check if the other two corners exist
                corner3 = (x1, y2)
                corner4 = (x2, y1)
                
                if corner3 in points_set and corner4 in points_set:
                    # Calculate area
                    area = abs(x1 - x2) * abs(y1 - y2)
                    
                    # Check if rectangle contains any other points
                    valid = True
                    for x, y in points_set:
                        # Skip the 4 corners
                        if (x, y) in [(x1, y1), (x2, y2), corner3, corner4]:
                            continue
                        
                        # Check if point is inside or on border
                        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
                            valid = False
                            break
                    
                    if valid and area > max_area:
                        max_area = area
        
        return max_area