from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum = sum(nums1)
        if total_sum <= x:
            return 0
        
        # Create a list of pairs (nums2[i], nums1[i])
        pairs = sorted(zip(nums2, nums1))
        
        # dp[k] will store the maximum reduction we can achieve in k operations
        dp = [0] * (n + 1)
        
        for i in range(n):
            for k in range(i + 1, 0, -1):
                dp[k] = max(dp[k], dp[k - 1] + pairs[i][0] * k + pairs[i][1])
        
        for k in range(n + 1):
            if total_sum - dp[k] <= x:
                return k
        
        return -1