class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        
        # Sort intervals by end time, keeping track of original indices
        sorted_indices = list(range(n))
        sorted_indices.sort(key=lambda i: intervals[i][1])
        
        # For each interval in sorted order, find the last compatible interval
        last_compatible = [-1] * n
        for i in range(n):
            idx_i = sorted_indices[i]
            # Binary search would be more efficient, but linear search is simpler
            for j in range(i - 1, -1, -1):
                idx_j = sorted_indices[j]
                if intervals[idx_j][1] < intervals[idx_i][0]:
                    last_compatible[i] = j
                    break
        
        # DP: dp[i][k] = (max_weight, indices_tuple)
        # considering first i sorted intervals with at most k selected
        INF = float('inf')
        dp = [[(-INF, None) for _ in range(5)] for _ in range(n + 1)]
        
        # Base case: selecting 0 intervals gives weight 0
        for i in range(n + 1):
            dp[i][0] = (0, ())
        
        # Fill DP table
        for i in range(1, n + 1):
            for k in range(1, 5):
                # Option 1: Don't take interval at sorted position i-1
                dp[i][k] = dp[i-1][k]
                
                # Option 2: Take interval at sorted position i-1
                idx = sorted_indices[i-1]
                weight = intervals[idx][2]
                
                # Find the best solution before this interval
                if last_compatible[i-1] == -1:
                    prev_weight, prev_indices = dp[0][k-1]
                else:
                    prev_weight, prev_indices = dp[last_compatible[i-1] + 1][k-1]
                
                if prev_indices is not None:
                    new_weight = prev_weight + weight
                    # Keep indices sorted for lexicographic comparison
                    new_indices = tuple(sorted(list(prev_indices) + [idx]))
                    
                    # Update if we found a better solution
                    if new_weight > dp[i][k][0] or \
                       (new_weight == dp[i][k][0] and (dp[i][k][1] is None or new_indices < dp[i][k][1])):
                        dp[i][k] = (new_weight, new_indices)
        
        # Find the best result across all possible numbers of intervals (1 to 4)
        best_weight = -INF
        best_indices = None
        
        for k in range(1, 5):
            weight, indices = dp[n][k]
            if indices is not None:
                if weight > best_weight or (weight == best_weight and (best_indices is None or indices < best_indices)):
                    best_weight = weight
                    best_indices = indices
        
        return list(best_indices) if best_indices else []