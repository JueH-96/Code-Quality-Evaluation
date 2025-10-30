import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Preprocess intervals with their original indices
        intervals_with_idx = [ (interval[0], interval[1], interval[2], idx) for idx, interval in enumerate(intervals) ]
        # Sort intervals by end time
        intervals_with_idx.sort(key=lambda x: (x[1], x[0], x[3]))
        n = len(intervals_with_idx)
        if n == 0:
            return []
        
        # Precompute p[i]
        ends = [x[1] for x in intervals_with_idx]
        p = []
        for i in range(n):
            l_i = intervals_with_idx[i][0]
            # Find the largest j where ends[j] <= l_i
            j = bisect.bisect_right(ends, l_i, 0, i) - 1
            p.append(j)
        
        # Initialize DP table
        dp = [ [ ( -float('inf'), [] ) for _ in range(n+1) ] for _ in range(5) ]
        dp[0][0] = (0, [])
        
        for i in range(1, n+1):
            current = intervals_with_idx[i-1]
            l, r, weight, idx = current
            # Copy previous state
            for m in range(5):
                dp[m][i] = dp[m][i-1]
            # Update for taking the current interval
            for m in range(4, 0, -1):
                # Find the previous state
                j = p[i-1]  # since i is 1-based here, intervals_with_idx[i-1] is the current interval
                if j == -1:
                    # Can't take any previous intervals
                    prev_sum = 0 if (m-1) == 0 else -float('inf')
                    prev_indices = []
                else:
                    prev_sum, prev_indices = dp[m-1][j+1]
                if prev_sum == -float('inf'):
                    continue  # no valid selection for m-1
                new_sum = prev_sum + weight
                new_indices = prev_indices + [idx]
                new_indices = sorted(new_indices)
                # Compare with current dp[m][i]
                current_sum, current_indices = dp[m][i]
                if new_sum > current_sum:
                    dp[m][i] = (new_sum, new_indices)
                elif new_sum == current_sum:
                    if new_indices < current_indices:
                        dp[m][i] = (new_sum, new_indices)
        
        # Find the best across all m
        max_sum = -float('inf')
        best_indices = []
        for m in range(5):
            current_sum, current_indices = dp[m][n]
            if current_sum > max_sum:
                max_sum = current_sum
                best_indices = current_indices
            elif current_sum == max_sum:
                if current_indices < best_indices:
                    best_indices = current_indices
        return best_indices