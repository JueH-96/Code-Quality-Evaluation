class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        
        point_set = set((x, y) for x, y in points)
        max_area = -1
        n = len(points)
        
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 == x2 or y1 == y2:
                    continue  # Skip if same x or y, can't form diagonal
                
                # Calculate the other two corners
                p3 = (x1, y2)
                p4 = (x2, y1)
                
                if p3 in point_set and p4 in point_set:
                    # Determine the rectangle's boundaries
                    x_min = min(x1, x2)
                    x_max = max(x1, x2)
                    y_min = min(y1, y2)
                    y_max = max(y1, y2)
                    
                    # Collect all four corners in a set for quick lookup
                    corners = {(x1, y1), (x2, y2), p3, p4}
                    valid = True
                    
                    # Check all points to ensure none are inside or on the rectangle's borders (except corners)
                    for p in points:
                        px, py = p
                        pt = (px, py)
                        if pt not in corners:
                            if x_min <= px <= x_max and y_min <= py <= y_max:
                                valid = False
                                break
                                
                    if valid:
                        current_area = (x_max - x_min) * (y_max - y_min)
                        if current_area > max_area:
                            max_area = current_area
        
        return max_area