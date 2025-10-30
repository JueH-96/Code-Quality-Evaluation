from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Transform points to two new coordinate systems
        # x' = x + y and y' = x - y
        transformed = [(x + y, x - y) for x, y in points]
        
        # Find the min and max in both transformed coordinate systems
        min_x_prime = min(transformed, key=lambda p: p[0])
        max_x_prime = max(transformed, key=lambda p: p[0])
        min_y_prime = min(transformed, key=lambda p: p[1])
        max_y_prime = max(transformed, key=lambda p: p[1])
        
        # Calculate the maximum distance without removing any point
        max_dist = max(manhattan_distance(min_x_prime, max_x_prime), manhattan_distance(min_y_prime, max_y_prime))
        
        # Try removing each point and calculate the new maximum distance
        result = float('inf')
        for i in range(len(points)):
            # Recalculate the min and max excluding the i-th point
            new_min_x_prime = min(transformed[:i] + transformed[i+1:], key=lambda p: p[0])
            new_max_x_prime = max(transformed[:i] + transformed[i+1:], key=lambda p: p[0])
            new_min_y_prime = min(transformed[:i] + transformed[i+1:], key=lambda p: p[1])
            new_max_y_prime = max(transformed[:i] + transformed[i+1:], key=lambda p: p[1])
            
            # Calculate the new maximum distance
            new_max_dist = max(manhattan_distance(new_min_x_prime, new_max_x_prime), manhattan_distance(new_min_y_prime, new_max_y_prime))
            result = min(result, new_max_dist)
        
        return result