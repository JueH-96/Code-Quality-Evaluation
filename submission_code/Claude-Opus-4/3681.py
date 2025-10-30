class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        from itertools import combinations
        
        n = len(points)
        if n < 4:
            return -1
        
        max_area = -1
        
        # Try all combinations of 4 points
        for combo in combinations(range(n), 4):
            # Get the 4 points
            rect_points = [points[i] for i in combo]
            
            # Extract x and y coordinates
            x_coords = [p[0] for p in rect_points]
            y_coords = [p[1] for p in rect_points]
            
            # Check if they form a rectangle parallel to axes
            # We need exactly 2 unique x values and 2 unique y values
            unique_x = sorted(set(x_coords))
            unique_y = sorted(set(y_coords))
            
            if len(unique_x) != 2 or len(unique_y) != 2:
                continue
            
            # Check if we have all 4 corners
            corners = set()
            for x in unique_x:
                for y in unique_y:
                    corners.add((x, y))
            
            rect_set = set((p[0], p[1]) for p in rect_points)
            if corners != rect_set:
                continue
            
            # Now we have a valid rectangle, check if any other point is inside or on border
            x1, x2 = unique_x[0], unique_x[1]
            y1, y2 = unique_y[0], unique_y[1]
            
            valid = True
            for i in range(n):
                if i in combo:
                    continue
                
                px, py = points[i]
                # Check if point is inside or on the border of rectangle
                if x1 <= px <= x2 and y1 <= py <= y2:
                    valid = False
                    break
            
            if valid:
                area = (x2 - x1) * (y2 - y1)
                max_area = max(max_area, area)
        
        return max_area