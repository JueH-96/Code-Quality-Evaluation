class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        intervals.sort(key=lambda x: x[1])  # Sort by end time
        
        ends = [interval[1] for interval in intervals]
        dp = [0] * (n + 1)
        prev = [0] * (n + 1)
        
        def binary_search(target, left, right):
            while left < right:
                mid = (left + right) // 2
                if ends[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left
        
        for i in range(1, n + 1):
            j = binary_search(intervals[i-1][0], 0, i-1)
            if dp[i-1] > dp[j] + intervals[i-1][2]:
                dp[i] = dp[i-1]
                prev[i] = prev[i-1]
            else:
                dp[i] = dp[j] + intervals[i-1][2]
                prev[i] = j
        
        result = []
        i = n
        while len(result) < 4 and i > 0:
            if prev[i] != i:
                result.append(i-1)
                i = prev[i]
            else:
                i -= 1
        
        return sorted(result)