import bisect
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by end, then start, then original index
        sorted_intervals = sorted(
            [(i, interval[0], interval[1], interval[2]) for i, interval in enumerate(intervals)],
            key=lambda x: (x[2], x[1], x[0])
        )
        n = len(sorted_intervals)
        ends = [x[2] for x in sorted_intervals]
        
        # Initialize DP: dp[k] = (max_weight, array_of_indices)
        INF = float('-inf')
        dp = [(INF, None) for _ in range(5)]
        dp[0] = (0, [])
        
        for i in range(n):
            original_idx, start, end, weight = sorted_intervals[i]
            # Find previous non-overlapping interval
            x = start
            pos = bisect.bisect_left(ends, x, 0, i)
            prev_idx = pos - 1
            
            # Process for k from 4 down to 1
            for k in range(4, 0, -1):
                prev_k = k - 1
                if prev_idx == -1:
                    if prev_k != 0:
                        continue
                    possible_weight = weight
                    possible_array = [original_idx]
                else:
                    if dp[prev_k][0] == INF:
                        continue
                    possible_weight = dp[prev_k][0] + weight
                    possible_array = dp[prev_k][1] + [original_idx]
                
                # Update dp[k]
                current_max, current_arr = dp[k]
                if possible_weight > current_max:
                    dp[k] = (possible_weight, possible_array)
                elif possible_weight == current_max:
                    if current_arr is None:
                        dp[k] = (possible_weight, possible_array)
                    else:
                        if possible_array < current_arr:
                            dp[k] = (possible_weight, possible_array)
        
        # Find the best array among dp[1..4]
        max_weight = INF
        best_arrays = []
        for k in range(1, 5):
            if dp[k][0] > max_weight:
                max_weight = dp[k][0]
                best_arrays = [dp[k][1]]
            elif dp[k][0] == max_weight:
                best_arrays.append(dp[k][1])
        
        # Now select the lex smallest array from best_arrays
        best_array = None
        for arr in best_arrays:
            if best_array is None:
                best_array = arr
            else:
                # Compare lengths first
                if len(arr) < len(best_array):
                    best_array = arr
                elif len(arr) == len(best_array):
                    if arr < best_array:
                        best_array = arr
        return best_array