class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        from itertools import combinations
        
        n = len(points)
        if n < 4:
            return -1
        
        max_area = -1
        
        # Try all combinations of 4 points
        for four_points in combinations(points, 4):
            # Check if these 4 points can form a rectangle
            if self.canFormRectangle(four_points):
                # Get rectangle bounds
                xs = [p[0] for p in four_points]
                ys = [p[1] for p in four_points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # Check if any other point lies inside or on the border
                valid = True
                for point in points:
                    if point not in four_points:
                        x, y = point
                        # Check if point is inside or on the border of rectangle
                        if min_x <= x <= max_x and min_y <= y <= max_y:
                            valid = False
                            break
                
                if valid:
                    area = (max_x - min_x) * (max_y - min_y)
                    max_area = max(max_area, area)
        
        return max_area
    
    def canFormRectangle(self, four_points):
        # For 4 points to form a rectangle with edges parallel to axes,
        # we need exactly 2 unique x-coordinates and 2 unique y-coordinates
        # and each combination of (x,y) should exist
        
        xs = [p[0] for p in four_points]
        ys = [p[1] for p in four_points]
        
        unique_xs = list(set(xs))
        unique_ys = list(set(ys))
        
        # Must have exactly 2 unique x and 2 unique y coordinates
        if len(unique_xs) != 2 or len(unique_ys) != 2:
            return False
        
        # Check if all 4 corner combinations exist
        expected_points = set()
        for x in unique_xs:
            for y in unique_ys:
                expected_points.add((x, y))
        
        actual_points = set((p[0], p[1]) for p in four_points)
        
        return expected_points == actual_points