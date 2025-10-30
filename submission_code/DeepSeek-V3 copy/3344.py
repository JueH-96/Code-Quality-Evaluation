class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n = len(points)
        if n == 3:
            return min(manhattan(points[0], points[1]), manhattan(points[1], points[2]), manhattan(points[0], points[2]))
        
        # Precompute the four extreme points
        x_plus_y = [p[0] + p[1] for p in points]
        x_minus_y = [p[0] - p[1] for p in points]
        
        max_x_plus_y = max(x_plus_y)
        min_x_plus_y = min(x_plus_y)
        max_x_minus_y = max(x_minus_y)
        min_x_minus_y = min(x_minus_y)
        
        # Find the indices of the extreme points
        max_x_plus_y_idx = x_plus_y.index(max_x_plus_y)
        min_x_plus_y_idx = x_plus_y.index(min_x_plus_y)
        max_x_minus_y_idx = x_minus_y.index(max_x_minus_y)
        min_x_minus_y_idx = x_minus_y.index(min_x_minus_y)
        
        # Collect all candidate points to remove
        candidates = set([max_x_plus_y_idx, min_x_plus_y_idx, max_x_minus_y_idx, min_x_minus_y_idx])
        
        min_max_distance = float('inf')
        
        for idx in candidates:
            # Create a new list without the current candidate
            new_points = points[:idx] + points[idx+1:]
            
            # Recompute the four extreme points for the new list
            new_x_plus_y = [p[0] + p[1] for p in new_points]
            new_x_minus_y = [p[0] - p[1] for p in new_points]
            
            new_max_x_plus_y = max(new_x_plus_y)
            new_min_x_plus_y = min(new_x_plus_y)
            new_max_x_minus_y = max(new_x_minus_y)
            new_min_x_minus_y = min(new_x_minus_y)
            
            # Calculate the maximum distance in the new list
            current_max_distance = max(new_max_x_plus_y - new_min_x_plus_y, new_max_x_minus_y - new_min_x_minus_y)
            
            if current_max_distance < min_max_distance:
                min_max_distance = current_max_distance
        
        return min_max_distance