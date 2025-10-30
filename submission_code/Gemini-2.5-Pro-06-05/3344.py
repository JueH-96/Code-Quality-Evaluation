import math
from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        """
        Calculates the minimum possible value for the maximum Manhattan distance
        between any two points after removing exactly one point.
        """
        n = len(points)
        
        # Step 1: Find indices of extreme points in u=x+y and v=x-y dimensions.
        # In a single O(N) pass, we find the points with the minimum/maximum u and v values.
        min_u, max_u = math.inf, -math.inf
        min_v, max_v = math.inf, -math.inf
        min_u_idx, max_u_idx, min_v_idx, max_v_idx = -1, -1, -1, -1

        for i, (x, y) in enumerate(points):
            u = x + y
            v = x - y
            
            if u < min_u:
                min_u, min_u_idx = u, i
            if u > max_u:
                max_u, max_u_idx = u, i
            if v < min_v:
                min_v, min_v_idx = v, i
            if v > max_v:
                max_v, max_v_idx = v, i

        # Step 2: Identify candidate indices to remove.
        # These are the indices of the four extreme points. The set will have 1 to 4 elements.
        candidate_indices = {min_u_idx, max_u_idx, min_v_idx, max_v_idx}
        
        min_of_max_distances = math.inf

        # Step 3: For each candidate, calculate max distance if it's removed.
        # Only removing an extreme point can potentially reduce the max distance.
        for idx_to_remove in candidate_indices:
            # Find the new extremes for u and v among the remaining N-1 points.
            temp_min_u, temp_max_u = math.inf, -math.inf
            temp_min_v, temp_max_v = math.inf, -math.inf
            
            for i in range(n):
                if i == idx_to_remove:
                    continue
                x, y = points[i]
                u, v = x + y, x - y
                temp_min_u = min(temp_min_u, u)
                temp_max_u = max(temp_max_u, u)
                temp_min_v = min(temp_min_v, v)
                temp_max_v = max(temp_max_v, v)
            
            # The maximum distance is the max of the new u and v ranges.
            current_max_dist = max(temp_max_u - temp_min_u, temp_max_v - temp_min_v)
            min_of_max_distances = min(min_of_max_distances, current_max_dist)
            
        return min_of_max_distances