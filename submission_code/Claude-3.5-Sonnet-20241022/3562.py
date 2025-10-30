class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # Add index to each interval and sort by end time
        intervals = [(i, l, r, w) for i, (l, r, w) in enumerate(intervals)]
        intervals.sort(key=lambda x: x[2])  # Sort by end time
        
        # Binary search to find the rightmost non-overlapping interval
        def binary_search(end_times, target, left):
            right = len(end_times) - 1
            result = -1
            
            while left <= right:
                mid = (left + right) // 2
                if end_times[mid] < target:
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            return result
        
        # dp[i][j] represents max weight using first i intervals and choosing j intervals
        dp = [[0] * 5 for _ in range(n + 1)]
        # prev[i][j] stores the previous interval index for reconstruction
        prev = [[None] * 5 for _ in range(n + 1)]
        
        end_times = [intervals[0][2]]  # List of end times for binary search
        for i in range(1, n):
            end_times.append(max(end_times[-1], intervals[i][2]))
        
        for i in range(1, n + 1):
            for j in range(1, min(5, i + 1)):
                # Don't take current interval
                dp[i][j] = dp[i-1][j]
                prev[i][j] = prev[i-1][j]
                
                # Try to take current interval
                idx, start, end, weight = intervals[i-1]
                prev_idx = binary_search(end_times, start, 0)
                
                curr_weight = weight
                if prev_idx >= 0:
                    curr_weight += dp[prev_idx + 1][j-1]
                
                if curr_weight > dp[i][j]:
                    dp[i][j] = curr_weight
                    prev[i][j] = (prev_idx + 1, j-1, idx)
        
        # Find maximum weight and number of intervals used
        max_weight = 0
        max_k = 0
        for j in range(5):
            if dp[n][j] > max_weight:
                max_weight = dp[n][j]
                max_k = j
        
        # Reconstruct solution
        result = []
        curr = n
        k = max_k
        while k > 0 and curr > 0:
            if prev[curr][k] is not None:
                prev_pos, prev_k, idx = prev[curr][k]
                if prev_pos != curr:
                    result.append(idx)
                curr = prev_pos
                k = prev_k
            else:
                curr -= 1
        
        # Sort result for lexicographically smallest answer
        result.sort()
        return result