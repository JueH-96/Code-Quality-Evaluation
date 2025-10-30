from typing import List
import math

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        # Constraint: 3 <= points.length

        # The Manhattan distance between (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|.
        # This can be rewritten as max(|(x1 + y1) - (x2 + y2)|, |(x1 - y1) - (x2 - y2)|).
        # Let u = x + y and v = x - y. The distance is max(|u1 - u2|, |v1 - v2|).
        # The maximum distance among a set of points is max(max(u) - min(u), max(v) - min(v))
        # over that set.

        # 1. Compute u = x + y and v = x - y for all points
        # Store (value, original_index) pairs for sorting
        # O(N) time and O(N) space
        u_values = [(p[0] + p[1], i) for i, p in enumerate(points)]
        v_values = [(p[0] - p[1], i) for i, p in enumerate(points)]

        # 2. Sort the lists based on the values
        # Sorting takes O(N log N) time.
        u_values.sort()
        v_values.sort()

        # Identify the indices of the points corresponding to the overall
        # minimum and maximum u and v values in the original set.
        # Removing one of these points is most likely to reduce the overall
        # maximum distance.
        # O(1) time.
        umin_idx = u_values[0][1]
        umax_idx = u_values[-1][1]
        vmin_idx = v_values[0][1]
        vmax_idx = v_values[-1][1]

        # The set of candidate points to remove are those achieving the
        # min/max u or v values. There are at most 4 such distinct points.
        # O(1) time and O(1) space.
        critical_indices = {umin_idx, umax_idx, vmin_idx, vmax_idx}

        # Initialize the minimum maximum distance found so far to infinity.
        min_max_dist = float('inf')

        # 3. Iterate through each critical index to consider removing the corresponding point
        # This loop runs at most 4 times. O(1) iterations.
        for k in critical_indices:
            # Calculate the min/max u and v values among the remaining points (P \ {p_k}).
            # This calculation takes O(1) time using the pre-sorted lists.
            
            # Determine the new min u value after removing point k.
            # If point k was the unique point with the overall minimum u value,
            # the new minimum u value is the second smallest u value.
            # Otherwise, the new minimum u value is the original smallest u value.
            # The sorted list `u_values` handles duplicates correctly: if the original
            # minimum u value was shared by multiple points, `u_values[0][0] == u_values[1][0]`.
            # If point k is `u_values[0][1]`, the new minimum is `u_values[1][0]`, which is
            # the same value as the original minimum, reflecting that other points still
            # have the minimum value.
            if u_values[0][1] == k:
                u_prime_min_val = u_values[1][0]
            else:
                u_prime_min_val = u_values[0][0]
                
            # Determine the new max u value after removing point k.
            # If point k was the unique point with the overall maximum u value,
            # the new maximum u value is the second largest u value.
            # Otherwise, the new maximum u value is the original largest u value.
            if u_values[-1][1] == k:
                u_prime_max_val = u_values[-2][0]
            else:
                u_prime_max_val = u_values[-1][0]

            # Determine the new min v value after removing point k.
            # Similar logic as for min u.
            if v_values[0][1] == k:
                v_prime_min_val = v_values[1][0]
            else:
                v_prime_min_val = v_values[0][0]

            # Determine the new max v value after removing point k.
            # Similar logic as for max u.
            if v_values[-1][1] == k:
                 v_prime_max_val = v_values[-2][0]
            else:
                v_prime_max_val = v_values[-1][0]
                
            # Calculate the maximum distance among the remaining points (P \ {p_k})
            # The maximum Manhattan distance is the maximum of the range of u values
            # and the range of v values among the remaining points.
            # O(1) time.
            current_max_dist = max(u_prime_max_val - u_prime_min_val, v_prime_max_val - v_prime_min_val)
            
            # Update the minimum maximum distance found so far among critical removals.
            # O(1) time.
            min_max_dist = min(min_max_dist, current_max_dist)

        # The maximum distance achieved by removing any non-critical point is the same
        # as the maximum distance among the original set of points (before removing any point).
        # This original maximum distance must also be considered, as the minimum overall
        # maximum distance could be achieved by removing a non-critical point (in which
        # case the minimum value would be the original maximum distance).
        # O(1) time.
        original_max_dist = max(u_values[-1][0] - u_values[0][0], v_values[-1][0] - v_values[0][0])
        
        # The final answer is the minimum of the maximum distances obtained by
        # removing a critical point (tracked in min_max_dist from the loop) and
        # the maximum distance obtained by removing a non-critical point (which is
        # the original_max_dist).
        # O(1) time.
        return min(min_max_dist, original_max_dist)