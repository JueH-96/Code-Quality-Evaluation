import bisect
from typing import List
import math

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # Add original index and sort by end time ascending, then original index ascending
        sorted_intervals = [(interval[0], interval[1], interval[2], idx) for idx, interval in enumerate(intervals)]
        sorted_intervals.sort(key=lambda x: (x[1], x[3]))
        
        # Extract end times for binary search
        end_list = [interval[1] for interval in sorted_intervals]
        
        # Compute prev for each position in sorted intervals
        prev = [-1] * n
        for pos in range(n):
            start_val = sorted_intervals[pos][0]
            P = bisect.bisect_left(end_list, start_val)
            min_val = min(pos, P)
            if min_val > 0:
                prev_j = min_val - 1
                prev[pos] = prev_j
        
        # DP setup: dp[i][c] stores (weight, sorted indices tuple)
        dp = [[None for _ in range(5)] for _ in range(n + 1)]
        
        # Initialize dp[0]
        for c in range(5):
            if c == 0:
                dp[0][c] = (0, ())  # weight and empty tuple
            else:
                dp[0][c] = (float('-inf'), None)
        
        # Fill DP table
        for i in range(1, n + 1):  # i is dp index, considering up to interval i (interval index i-1)
            pos = i - 1  # position in sorted intervals
            weight_interval = sorted_intervals[pos][2]
            orig_idx_interval = sorted_intervals[pos][3]
            
            for c in range(5):  # c from 0 to 4
                if c == 0:
                    # For c=0, always no intervals chosen
                    dp[i][c] = (0, ())
                    continue
                
                # Not take option
                not_take_weight, not_take_indices = dp[i-1][c]
                
                # Take option
                prev_j = prev[pos]
                if prev_j >= 0:
                    prev_dp_i_val = prev_j + 1
                else:
                    prev_dp_i_val = 0
                prev_dp_weight, prev_dp_indices = dp[prev_dp_i_val][c - 1]
                
                if prev_dp_weight != float('-inf'):
                    take_weight = weight_interval + prev_dp_weight
                    # Create new sorted indices tuple
                    new_indices_list = list(prev_dp_indices)
                    new_indices_list.append(orig_idx_interval)
                    new_indices_list.sort()
                    take_indices_tuple = tuple(new_indices_list)
                else:
                    take_weight = float('-inf')
                    take_indices_tuple = None
                
                # Compare options
                if take_weight > not_take_weight:
                    dp[i][c] = (take_weight, take_indices_tuple)
                elif take_weight < not_take_weight:
                    dp[i][c] = (not_take_weight, not_take_indices)
                else:  # weights equal
                    if take_weight == float('-inf'):  # both invalid
                        dp[i][c] = (not_take_weight, not_take_indices)  # choose not take
                    else:  # both finite and equal
                        if take_indices_tuple < not_take_indices:  # lexicographically smaller
                            dp[i][c] = (take_weight, take_indices_tuple)
                        else:
                            dp[i][c] = (not_take_weight, not_take_indices)
        
        # Find maximum weight at dp[n]
        dp_n = dp[n]
        W_max = max(dp_n[c][0] for c in range(5))
        
        # Find all candidates with weight W_max
        candidates = [dp_n[c][1] for c in range(5) if dp_n[c][0] == W_max]
        
        # Find the lexicographically smallest indices tuple
        min_indices_tuple = min(candidates)
        
        # Return as list
        return list(min_indices_tuple)