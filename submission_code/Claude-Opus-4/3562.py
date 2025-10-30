class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        
        # Add index to each interval for tracking
        indexed_intervals = [(intervals[i][0], intervals[i][1], intervals[i][2], i) 
                           for i in range(n)]
        
        # Sort by end time
        indexed_intervals.sort(key=lambda x: x[1])
        
        # For each interval, find the latest non-overlapping interval before it
        prev = [-1] * n
        for i in range(n):
            for j in range(i - 1, -1, -1):
                if indexed_intervals[j][1] < indexed_intervals[i][0]:
                    prev[i] = j
                    break
        
        # dp[i][j] = (max_weight, indices) for first i intervals using at most j intervals
        # dp[i][j][0] = maximum weight
        # dp[i][j][1] = list of original indices used
        dp = [[(-1, []) for _ in range(5)] for _ in range(n + 1)]
        
        # Base case
        for j in range(5):
            dp[0][j] = (0, [])
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(5):
                # Option 1: Don't take current interval
                dp[i][j] = dp[i-1][j]
                
                # Option 2: Take current interval (if j > 0)
                if j > 0:
                    curr_weight = indexed_intervals[i-1][2]
                    curr_idx = indexed_intervals[i-1][3]
                    
                    if prev[i-1] == -1:
                        # No previous non-overlapping interval
                        new_weight = curr_weight
                        new_indices = [curr_idx]
                    else:
                        # Add to previous non-overlapping solution
                        prev_weight, prev_indices = dp[prev[i-1] + 1][j-1]
                        new_weight = prev_weight + curr_weight
                        new_indices = prev_indices + [curr_idx]
                    
                    # Update if this gives better weight or same weight with lex smaller indices
                    if new_weight > dp[i][j][0] or (new_weight == dp[i][j][0] and 
                                                     sorted(new_indices) < sorted(dp[i][j][1])):
                        dp[i][j] = (new_weight, new_indices)
        
        # Find the best solution across all possible number of intervals (1 to 4)
        best_weight = -1
        best_indices = []
        
        for j in range(1, 5):
            weight, indices = dp[n][j]
            if weight > best_weight or (weight == best_weight and 
                                       sorted(indices) < sorted(best_indices)):
                best_weight = weight
                best_indices = indices
        
        return sorted(best_indices)