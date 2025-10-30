class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Create a set of points for quick lookup
        point_set = set((x, y) for x, y in points)
        
        # Dictionary to store x coordinates grouped by y
        x_by_y = defaultdict(list)
        for x, y in points:
            x_by_y[y].append(x)
        
        # Sort x values for each y for easier processing
        for y in x_by_y:
            x_by_y[y].sort()
        
        # Dictionary to store y coordinates grouped by x
        y_by_x = defaultdict(list)
        for x, y in points:
            y_by_x[x].append(y)
        
        # Sort y values for each x for easier processing
        for x in y_by_x:
            y_by_x[x].sort()
        
        max_area = -1
        
        # Process each pair of y-values to find potential rectangles
        sorted_ys = sorted(x_by_y.keys())
        n = len(sorted_ys)
        
        for i in range(n):
            for j in range(i + 1, n):
                y1 = sorted_ys[i]
                y2 = sorted_ys[j]
                
                # Find common x values for these two y levels
                common_xs = sorted(set(x_by_y[y1]) & set(x_by_y[y2]))
                
                # Now check for each pair of x in common_xs if it forms a valid rectangle
                for k in range(len(common_xs) - 1):
                    x1 = common_xs[k]
                    x2 = common_xs[k + 1]
                    
                    # Check if rectangle formed by (x1, y1), (x1, y2), (x2, y1), (x2, y2) is valid
                    if (x1, y1) in point_set and (x1, y2) in point_set and (x2, y1) in point_set and (x2, y2) in point_set:
                        area = (x2 - x1) * (y2 - y1)
                        max_area = max(max_area, area)
        
        return max_area