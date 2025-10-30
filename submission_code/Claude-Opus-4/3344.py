class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        
        # Calculate x+y and x-y for all points
        sum_coords = [(x + y, i) for i, (x, y) in enumerate(points)]
        diff_coords = [(x - y, i) for i, (x, y) in enumerate(points)]
        
        # Sort to easily find min and max
        sum_coords.sort()
        diff_coords.sort()
        
        # Find initial maximum distance
        max_sum_dist = sum_coords[-1][0] - sum_coords[0][0]
        max_diff_dist = diff_coords[-1][0] - diff_coords[0][0]
        
        # Identify critical points (those that contribute to max distance)
        critical_points = set()
        if max_sum_dist >= max_diff_dist:
            critical_points.add(sum_coords[0][1])
            critical_points.add(sum_coords[-1][1])
        if max_diff_dist >= max_sum_dist:
            critical_points.add(diff_coords[0][1])
            critical_points.add(diff_coords[-1][1])
        
        min_max_distance = float('inf')
        
        # Try removing each critical point
        for remove_idx in critical_points:
            # Find max distance after removing point at remove_idx
            # For sum coordinates
            sum_vals = [val for val, idx in sum_coords if idx != remove_idx]
            max_sum = max(sum_vals) - min(sum_vals) if sum_vals else 0
            
            # For diff coordinates
            diff_vals = [val for val, idx in diff_coords if idx != remove_idx]
            max_diff = max(diff_vals) - min(diff_vals) if diff_vals else 0
            
            # Maximum distance after removal
            max_dist = max(max_sum, max_diff)
            min_max_distance = min(min_max_distance, max_dist)
        
        return min_max_distance