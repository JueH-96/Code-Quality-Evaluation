class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        # Function to calculate the Manhattan distance between two points
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Precompute the four key points for the initial set
        def get_key_points(points):
            x_plus_y = [p[0] + p[1] for p in points]
            x_minus_y = [p[0] - p[1] for p in points]
            max_x_plus_y = max(x_plus_y)
            min_x_plus_y = min(x_plus_y)
            max_x_minus_y = max(x_minus_y)
            min_x_minus_y = min(x_minus_y)
            # Find the indices of these key points
            max_x_plus_y_idx = x_plus_y.index(max_x_plus_y)
            min_x_plus_y_idx = x_plus_y.index(min_x_plus_y)
            max_x_minus_y_idx = x_minus_y.index(max_x_minus_y)
            min_x_minus_y_idx = x_minus_y.index(min_x_minus_y)
            return {max_x_plus_y_idx, min_x_plus_y_idx, max_x_minus_y_idx, min_x_minus_y_idx}
        
        # Get the key points
        key_points = get_key_points(points)
        
        # Initialize the minimum maximum distance to a large number
        min_max_distance = float('inf')
        
        # Iterate over each key point to remove and compute the new maximum distance
        for idx in key_points:
            # Create a new list without the current key point
            new_points = points[:idx] + points[idx+1:]
            # Recompute the key points for the new list
            new_key_points = get_key_points(new_points)
            # Compute the maximum distance among the new key points
            max_distance = 0
            for i in new_key_points:
                for j in new_key_points:
                    if i != j:
                        distance = manhattan_distance(new_points[i], new_points[j])
                        if distance > max_distance:
                            max_distance = distance
            # Update the minimum maximum distance
            if max_distance < min_max_distance:
                min_max_distance = max_distance
        
        return min_max_distance