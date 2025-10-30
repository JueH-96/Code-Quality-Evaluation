from typing import List
import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Enumerate intervals with their original indices
        enumerated_intervals = [(interval[0], interval[1], interval[2], idx) for idx, interval in enumerate(intervals)]
        # Sort intervals by end time
        enumerated_intervals.sort(key=lambda x: x[1])
        
        # Initialize dp lists for k = 0 to 4
        # Each dp[k] is a list of tuples: (end_time, total_weight, indices_list)
        dp = [[] for _ in range(5)]  # dp[0] to dp[4]
        dp[0].append( ( -1, 0, [] ) )  # Dummy interval for k=0
        
        for interval in enumerated_intervals:
            start, end, weight, idx = interval
            # Iterate k from 4 down to 1 to avoid using updated dp[k-1] in the same iteration
            for k in range(4, 0, -1):
                if not dp[k-1]:
                    continue
                # Binary search to find the last interval in dp[k-1] with end_time < start
                # Since dp[k-1] is sorted by end_time, we can use bisect
                # We need to find rightmost end_time < start
                # Create a list of end_times for binary search
                end_times = [entry[0] for entry in dp[k-1]]
                idx_bisect = bisect.bisect_left(end_times, start) - 1
                if idx_bisect >= 0:
                    prev_end, prev_weight, prev_indices = dp[k-1][idx_bisect]
                    new_weight = prev_weight + weight
                    new_indices = prev_indices + [idx]
                    # Check if we need to update dp[k]
                    if not dp[k]:
                        dp[k].append( (end, new_weight, new_indices) )
                    else:
                        last_weight, last_indices = dp[k][-1][1], dp[k][-1][2]
                        if new_weight > last_weight or (new_weight == last_weight and new_indices < last_indices):
                            dp[k].append( (end, new_weight, new_indices) )
            
        # After processing all intervals, find the best among dp[1] to dp[4]
        best_weight = -1
        best_indices = []
        for k in range(1,5):
            if not dp[k]:
                continue
            # The last entry in dp[k] has the maximum weight for that k
            current_weight = dp[k][-1][1]
            current_indices = dp[k][-1][2]
            if current_weight > best_weight or (current_weight == best_weight and current_indices < best_indices):
                best_weight = current_weight
                best_indices = current_indices
        
        return best_indices