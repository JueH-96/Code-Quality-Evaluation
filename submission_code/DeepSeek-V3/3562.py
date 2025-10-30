from bisect import bisect_left

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals based on end time
        intervals_sorted = sorted([(l, r, w, i) for i, (l, r, w) in enumerate(intervals)], key=lambda x: x[1])
        n = len(intervals_sorted)
        
        # Preprocess to get the list of end times
        end_times = [x[1] for x in intervals_sorted]
        
        # Initialize DP arrays
        dp = [0] * (n + 1)
        prev = [0] * (n + 1)
        chosen = [[] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            l, r, w, idx = intervals_sorted[i-1]
            # Find the latest interval that does not overlap with the current one
            j = bisect_left(end_times, l)
            # Decide whether to take the current interval or not
            if dp[j] + w > dp[i-1]:
                dp[i] = dp[j] + w
                prev[i] = j
                chosen[i] = chosen[j] + [idx]
            else:
                dp[i] = dp[i-1]
                prev[i] = i-1
                chosen[i] = chosen[i-1]
        
        # Now, we need to find the maximum weight with at most 4 intervals
        # We can limit the number of intervals by checking the length of chosen[i]
        max_weight = 0
        best_indices = []
        for i in range(n + 1):
            if len(chosen[i]) <= 4 and dp[i] > max_weight:
                max_weight = dp[i]
                best_indices = chosen[i]
        
        # Sort the indices lexicographically
        best_indices.sort()
        return best_indices