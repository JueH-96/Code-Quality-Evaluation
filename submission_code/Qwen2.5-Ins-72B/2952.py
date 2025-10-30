from typing import List
import heapq

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum = sum(nums1)
        if total_sum <= x:
            return 0
        
        # Create a list of (nums2[i], nums1[i]) pairs and sort by nums2
        pairs = sorted((nums2[i], nums1[i]) for i in range(n))
        
        # dp[i][j] will be the maximum sum of nums1 after j operations on the first i pairs
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j]
                if j > 0:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + pairs[i - 1][1] + pairs[i - 1][0] * j)
        
        for t in range(n + 1):
            if total_sum + t * sum(nums2) - dp[n][t] <= x:
                return t
        
        return -1