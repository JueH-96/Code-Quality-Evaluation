from typing import List
import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by end time, keep track of original indices
        sorted_intervals = sorted([(interval[1], interval[0], interval[2], idx) for idx, interval in enumerate(intervals)], key=lambda x: x[0])
        
        # Extract sorted end times for binary search
        ends = [interval[0] for interval in sorted_intervals]
        
        n = len(intervals)
        k_max = 4
        
        # Initialize dp and predecessor arrays
        dp = [[0] * n for _ in range(k_max + 1)]
        predecessor = [[-1] * n for _ in range(k_max + 1)]
        
        # Fill dp table
        for i in range(n):
            start_i = sorted_intervals[i][1]
            weight_i = sorted_intervals[i][2]
            original_idx_i = sorted_intervals[i][3]
            
            # k=1
            dp[1][i] = weight_i
            predecessor[1][i] = -1
            
            # Find j for each k from 2 to 4
            for k in range(2, k_max + 1):
                # Find the latest interval j where end_j < start_i
                j = bisect.bisect_left(ends, start_i) - 1
                if j != -1:
                    if dp[k-1][j] + weight_i > dp[k][i]:
                        dp[k][i] = dp[k-1][j] + weight_i
                        predecessor[k][i] = j
                    elif dp[k-1][j] + weight_i == dp[k][i]:
                        # Choose the smaller j to get lex smallest indices
                        if predecessor[k][i] > j:
                            predecessor[k][i] = j
                else:
                    # Not possible to select k intervals, keep dp[k][i] as 0
                    dp[k][i] = 0
                    predecessor[k][i] = -1
        
        # Find the maximum weight and corresponding k and i
        max_weight = -1
        chosen_k = 0
        chosen_i = -1
        for k in range(1, k_max + 1):
            current_max = max(dp[k])
            if current_max > max_weight:
                max_weight = current_max
                chosen_k = k
                chosen_i = dp[k].index(current_max)
            elif current_max == max_weight:
                # Choose the smallest k
                if k < chosen_k:
                    chosen_k = k
                    chosen_i = dp[k].index(current_max)
        
        # Backtrack to find the selected intervals
        selected_sorted_indices = []
        current_k = chosen_k
        current_i = chosen_i
        while current_k > 0:
            selected_sorted_indices.append(current_i)
            current_i = predecessor[current_k][current_i]
            current_k -= 1
        selected_sorted_indices = selected_sorted_indices[::-1]
        
        # Collect original indices and sort them
        selected_original_indices = [sorted_intervals[i][3] for i in selected_sorted_indices]
        selected_original_indices.sort()
        
        return selected_original_indices