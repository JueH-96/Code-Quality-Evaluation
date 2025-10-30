import math
from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def manhattan_dist(p1: List[int], p2: List[int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # 1. Map points to perimeter values and sort them
        perim_points_with_val = []
        for p in points:
            x, y = p[0], p[1]
            p_val = 0
            if y == 0:  # Bottom edge (0,0) to (side,0)
                p_val = x
            elif x == side:  # Right edge (side,0) to (side,side)
                p_val = side + y
            elif y == side:  # Top edge (side,side) to (0,side)
                p_val = side + side + (side - x)
            elif x == 0:  # Left edge (0,side) to (0,0)
                p_val = side + side + side + (side - y)
            perim_points_with_val.append((p_val, p))
        
        perim_points_with_val.sort()
        
        actual_points = [p[1] for p in perim_points_with_val]
        N = len(actual_points)

        # check(D): Can we select k points such that min Manhattan distance is at least D?
        def check(D: int) -> bool:
            # Precompute next_idx[i]: the smallest j > i (in conceptual doubled array)
            # such that manhattan_dist(actual_points[j%N], actual_points[i%N]) >= D.
            # This is done using a two-pointer approach.
            next_idx = [math.inf] * (2 * N)
            j_ptr = 0 # Pointer for the "next" valid point candidate

            for i in range(2 * N):
                # Ensure j_ptr is always ahead of i. For i=0, j_ptr starts from 1.
                # For i > 0, j_ptr should start from at least i+1.
                if j_ptr <= i:
                    j_ptr = i + 1
                
                # Find the first point that satisfies the distance D
                while j_ptr < 2 * N and manhattan_dist(actual_points[j_ptr % N], actual_points[i % N]) < D:
                    j_ptr += 1
                
                next_idx[i] = j_ptr
            
            # DP to check if k points can be selected.
            # dp_val[c] stores the minimum index in the conceptual `doubled_points` array 
            # that can be the c-th selected point (1-indexed).
            
            # Iterate through each point in the original list as a potential starting point
            for start_node_original_idx in range(N):
                # dp_val[0] is unused. dp_val[1] to dp_val[k].
                dp_val = [math.inf] * (k + 1)
                
                # Base case: The first point is at start_node_original_idx
                dp_val[1] = start_node_original_idx
                
                # Fill DP table for c from 2 to k
                for c in range(2, k + 1):
                    prev_idx = dp_val[c-1]
                    if prev_idx != math.inf:
                        # The next possible point for the (c-1)-th point at prev_idx
                        # is found using the precomputed next_idx array.
                        dp_val[c] = next_idx[prev_idx]
                    # If prev_idx is inf, then dp_val[c] remains inf.
                
                # After filling DP, check if k points were successfully selected
                if dp_val[k] != math.inf:
                    # dp_val[k] is the index in the conceptual doubled array for the k-th point.
                    # This index must be within one full cycle from the start_node_original_idx.
                    # The condition `dp_val[k] < start_node_original_idx + N` ensures that:
                    # 1. The selected points form a sequence of `k` distinct points from the original `N` points.
                    #    (i.e., the last point doesn't wrap around to be one of the previously chosen points in `actual_points`).
                    # 2. The selected points are "forward" relative to the `start_node_original_idx` within a single cycle.
                    if dp_val[k] < start_node_original_idx + N:
                        # Finally, check the cyclic distance constraint:
                        # The distance between the last selected point and the first selected point.
                        last_point_orig_idx = dp_val[k] % N
                        first_point_orig_idx = start_node_original_idx % N
                        
                        if manhattan_dist(actual_points[last_point_orig_idx], actual_points[first_point_orig_idx]) >= D:
                            return True
            return False

        # Binary search for the maximum D
        low = 0
        # Maximum possible Manhattan distance on a square boundary is 2 * side (e.g., from (0,0) to (side,side))
        high = 2 * side 
        ans = 0

        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans