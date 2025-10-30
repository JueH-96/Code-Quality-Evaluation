import math
from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)

        # Step 1: Precompute u = x+y and v = x-y values for all points
        # Store them along with their original indices
        u_values_indexed = [] # List of (x+y, original_index)
        v_values_indexed = [] # List of (x-y, original_index)

        for i, (x, y) in enumerate(points):
            u_values_indexed.append((x + y, i))
            v_values_indexed.append((x - y, i))

        # Step 2: Sort the lists based on their values
        u_values_indexed.sort()
        v_values_indexed.sort()

        # Step 3: Find overall max/min u and v values and their indices
        # For u values
        overall_min_u_val, overall_min_u_idx = u_values_indexed[0]
        overall_max_u_val, overall_max_u_idx = u_values_indexed[-1]
        
        # Second smallest and second largest u values (for when an extreme point is removed)
        # N >= 3 is guaranteed by constraints, so these indices (1 and -2) are safe.
        second_min_u_val, _ = u_values_indexed[1]
        second_max_u_val, _ = u_values_indexed[-2]

        # For v values
        overall_min_v_val, overall_min_v_idx = v_values_indexed[0]
        overall_max_v_val, overall_max_v_idx = v_values_indexed[-1]

        # Second smallest and second largest v values
        second_min_v_val, _ = v_values_indexed[1]
        second_max_v_val, _ = v_values_indexed[-2]

        min_overall_max_dist = math.inf

        # Step 4: Iterate through each point to simulate its removal
        for k in range(n):
            # Determine new_max_u and new_min_u after removing points[k]
            new_max_u = overall_max_u_val
            new_min_u = overall_min_u_val

            # Check if the removed point 'k' was the one yielding the overall maximum 'u' value.
            # If it was, and this maximum 'u' value was unique (i.e., not equal to the second largest 'u' value),
            # then the new maximum 'u' becomes the second largest 'u' value.
            # Otherwise, if the overall max 'u' was not unique, or point 'k' was not the max 'u' point,
            # new_max_u remains the overall_max_u_val.
            if k == overall_max_u_idx:
                if u_values_indexed[-2][0] != overall_max_u_val:
                    new_max_u = second_max_u_val
            
            # Apply similar logic for determining new_min_u
            if k == overall_min_u_idx:
                if u_values_indexed[1][0] != overall_min_u_val:
                    new_min_u = second_min_u_val
            
            range_u = new_max_u - new_min_u

            # Determine new_max_v and new_min_v after removing points[k]
            new_max_v = overall_max_v_val
            new_min_v = overall_min_v_val

            if k == overall_max_v_idx:
                if v_values_indexed[-2][0] != overall_max_v_val:
                    new_max_v = second_max_v_val
            
            if k == overall_min_v_idx:
                if v_values_indexed[1][0] != overall_min_v_val:
                    new_min_v = second_min_v_val
            
            range_v = new_max_v - new_min_v

            # The maximum Manhattan distance for the remaining N-1 points
            current_max_dist = max(range_u, range_v)
            
            # Update the minimum overall maximum distance found so far
            min_overall_max_dist = min(min_overall_max_dist, current_max_dist)

        return min_overall_max_dist