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
            
            # Get all x and y coordinates
            xs = [p[0] for p in rect_points]
            ys = [p[1] for p in rect_points]
            
            # For a rectangle, there should be exactly 2 unique x values and 2 unique y values
            unique_xs = sorted(list(set(xs)))
            unique_ys = sorted(list(set(ys)))
            
            if len(unique_xs) != 2 or len(unique_ys) != 2:
                continue
            
            # Check if each unique x,y combination exists
            x1, x2 = unique_xs
            y1, y2 = unique_ys
            
            required_points = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
            actual_points = {(p[0], p[1]) for p in rect_points}
            
            if required_points != actual_points:
                continue
            
            # Now check if any other point lies inside or on the border
            valid = True
            for i in range(n):
                if i in combo:
                    continue
                
                x, y = points[i]
                # Check if point is inside or on border
                if x1 <= x <= x2 and y1 <= y <= y2:
                    valid = False
                    break
            
            if valid:
                area = (x2 - x1) * (y2 - y1)
                max_area = max(max_area, area)
        
        return max_area