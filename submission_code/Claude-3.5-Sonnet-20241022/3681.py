class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert points to set for O(1) lookup
        point_set = set(map(tuple, points))
        n = len(points)
        max_area = -1
        
        # Try all possible pairs of points as diagonal corners
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                # Skip if points are on same x or y coordinate
                if x1 == x2 or y1 == y2:
                    continue
                
                # Check if other two corners exist
                corner1 = (x1, y2)
                corner2 = (x2, y1)
                
                if corner1 in point_set and corner2 in point_set:
                    # Calculate area
                    area = abs(x2 - x1) * abs(y2 - y1)
                    
                    # Check if any point lies inside the rectangle
                    valid = True
                    min_x, max_x = min(x1, x2), max(x1, x2)
                    min_y, max_y = min(y1, y2), max(y1, y2)
                    
                    for x, y in points:
                        # Skip corner points
                        if (x, y) in [(x1, y1), (x2, y2), corner1, corner2]:
                            continue
                        # Check if point lies inside rectangle
                        if min_x < x < max_x and min_y < y < max_y:
                            valid = False
                            break
                    
                    if valid:
                        max_area = max(max_area, area)
        
        return max_area