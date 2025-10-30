from typing import List
from collections import defaultdict
import bisect

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Step 1: Create delta_map for C(x) values
        # c_delta_at_pos[p] stores the change in coin value at position p
        c_delta_at_pos = defaultdict(int)
        for l, r, c in coins:
            c_delta_at_pos[l] += c
            c_delta_at_pos[r + 1] -= c

        # Get sorted unique points where C(x) changes
        # Store as list of (point, delta) tuples for easier iteration
        points_and_deltas = sorted(c_delta_at_pos.items())

        # Handle case where coins list is empty
        if not points_and_deltas:
             return 0

        # Step 2: Create value_intervals [(start, end, value)] for non-zero value intervals
        # These intervals cover ranges [p_i, p_{i+1}-1] where C(x) is constant and non-zero.
        value_intervals = []
        current_c = 0 # Coin value for x < points_and_deltas[0][0]
        
        last_p = points_and_deltas[0][0] # The first point where coin value might change

        # Iterate through all points and their deltas
        for p, delta in points_and_deltas:
            # The interval [last_p, p - 1] exists if p > last_p.
            # The value within this interval is `current_c` *before* adding the delta at `p`.
            if p > last_p:
                # Only add intervals with non-zero coin value
                if current_c != 0:
                    value_intervals.append((last_p, p - 1, current_c))
            
            # Update current_c for positions >= p
            current_c += delta
            last_p = p # Update last_p for the next iteration

        # Helper function to query C(x) efficiently
        def query_C_helper(x: int, intervals: List[tuple]) -> int:
            # Find the index of the first interval starting strictly after x.
            # Search tuple (x, float('inf'), float('inf')) to ensure correct bisect_right behavior
            # Intervals are sorted by (start, end, value)
            # We want the interval [start, end] such that start <= x <= end.
            # bisect_right finds the index i such that intervals[i-1].start <= x < intervals[i].start
            idx = bisect.bisect_right(intervals, (x, float('inf'), float('inf')))

            # The interval potentially containing x is at index idx - 1
            if idx > 0:
                start, end, val = intervals[idx - 1]
                # Check if x falls within this interval's range [start, end]
                if start <= x <= end:
                    return val
            return 0 # C(x) is 0 outside defined intervals

        # Step 3: Create rate_delta_map for changes in Rate(s) = C(s+k) - C(s)
        rate_delta_map = defaultdict(int)
        # Events happen when s or s+k crosses a point where C(x) changes (i.e., points in c_delta_at_pos)
        # Use the original c_delta_at_pos dictionary keys for sweep points
        for p, delta_c in c_delta_at_pos.items():
            # When s crosses p: C(s) changes by delta_c. Rate changes by -delta_c.
            rate_delta_map[p] -= delta_c
            # When s+k crosses p (i.e., s crosses p-k): C(s+k) changes by delta_c. Rate changes by +delta_c.
            # Ensure p - k is included as a potential sweep point
            rate_delta_map[p - k] += delta_c

        # Get sorted unique sweep points for s
        sweep_points = sorted(rate_delta_map.keys())

        # Handle case where sweep_points is empty (shouldn't happen if coins is not empty)
        if not sweep_points:
            return 0 # Should be unreachable if points_and_deltas is not empty

        # Step 5 & 6: Find the smallest sweep point s_0 and calculate initial sum F(s_0)
        s_0 = sweep_points[0]
        current_sum = 0
        window_end_s0 = s_0 + k - 1

        # Calculate F(s_0) = Sum_{x=s_0}^{s_0+k-1} C(x)
        # Iterate through relevant value_intervals that might overlap with [s_0, window_end_s0]
        
        # Find the index of the first interval whose start is strictly after window_end_s0
        # We only need to consider intervals [a, b] where a <= window_end_s0
        first_idx_after_window = bisect.bisect_right(value_intervals, (window_end_s0, float('inf'), float('inf')))
        
        # Iterate through intervals [a, b, v] where a <= window_end_s0
        for i in range(first_idx_after_window):
            a, b, v = value_intervals[i]
            
            # Check if the interval [a, b] actually overlaps with [s_0, window_end_s0], i.e., b >= s_0
            if b >= s_0:
                overlap_start = max(s_0, a)
                overlap_end = min(window_end_s0, b)

                # Since we are in the loop (a <= window_end_s0) and checked b >= s_0,
                # and intervals have start <= end, we are guaranteed overlap_start <= overlap_end.
                overlap_length = overlap_end - overlap_start + 1
                current_sum += v * overlap_length
                
        # Step 7: Calculate initial rate Rate(s_0) = C(s_0 + k) - C(s_0)
        current_rate = query_C_helper(s_0 + k, value_intervals) - query_C_helper(s_0, value_intervals)

        # Step 8: Initialize max_sum
        max_sum = current_sum

        # Step 9: Sweep through sweep_points from the second point onwards
        last_p = s_0
        for i in range(1, len(sweep_points)):
            p = sweep_points[i]

            # The sum changes linearly from last_p to p-1 with the current_rate
            # F(p) = F(last_p) + Rate(last_p) * (p - last_p)
            # Check p > last_p just in case, though sorted unique points guarantee this for i > 0
            if p > last_p:
                 current_sum += current_rate * (p - last_p)

                 # Update max_sum at the new point p
                 max_sum = max(max_sum, current_sum)

            # Update current_rate for the interval starting at p
            # This update applies to the rate for s >= p
            current_rate += rate_delta_map[p]

            # Move to the next point
            last_p = p

        return max_sum