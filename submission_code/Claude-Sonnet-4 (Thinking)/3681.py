class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        from itertools import combinations
        
        if len(points) < 4:
            return -1
        
        point_set = set(map(tuple, points))
        max_area = -1
        
        # Try all pairs of points as potential diagonal corners
        for p1, p2 in combinations(points, 2):
            x1, y1 = p1
            x2, y2 = p2
            
            # Check if they can be diagonal corners (different x and y)
            if x1 == x2 or y1 == y2:
                continue
            
            # The other two corners needed for the rectangle
            corner3 = (x1, y2)
            corner4 = (x2, y1)
            
            # Check if the other two corners exist in our point set
            if corner3 not in point_set or corner4 not in point_set:
                continue
            
            # Define the rectangle boundaries
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            
            # The four corners of this rectangle
            rectangle_corners = {(x1, y1), (x2, y2), corner3, corner4}
            
            # Check if any other point is inside or on the border
            valid = True
            for point in points:
                px, py = point
                if (px, py) in rectangle_corners:
                    continue
                
                # Check if point is inside or on border of rectangle
                if min_x <= px <= max_x and min_y <= py <= max_y:
                    valid = False
                    break
            
            if valid:
                area = (max_x - min_x) * (max_y - min_y)
                max_area = max(max_area, area)
        
        return max_area