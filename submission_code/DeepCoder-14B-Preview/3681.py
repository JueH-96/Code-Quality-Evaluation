class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        max_area = -1
        # Convert points to tuples for easier handling
        points = [tuple(point) for point in points]
        points_set = set(points)
        n = len(points)
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                x1, y1 = p1
                x2, y2 = p2
                
                # Skip if they can't form a diagonal
                if x1 == x2 or y1 == y2:
                    continue
                
                # Calculate the other two points needed for the rectangle
                p3 = (x1, y2)
                p4 = (x2, y1)
                
                # Check if both other points exist
                if p3 in points_set and p4 in points_set:
                    left = min(x1, x2)
                    right = max(x1, x2)
                    bottom = min(y1, y2)
                    top = max(y1, y2)
                    
                    # Check all other points to ensure none are inside or on the border
                    valid = True
                    for point in points:
                        if point not in (p1, p2, p3, p4):
                            x, y = point
                            if left <= x <= right and bottom <= y <= top:
                                valid = False
                                break
                    
                    if valid:
                        area = (right - left) * (top - bottom)
                        if area > max_area:
                            max_area = area
        
        return max_area if max_area != -1 else -1