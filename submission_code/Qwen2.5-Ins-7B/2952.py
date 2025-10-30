from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_increase = sum(nums2)
        dp = [0] * (n + 1)
        nums1_sum = sum(nums1)
        
        for i in range(n):
            nums1[i] += nums2[i]
        
        for i in range(n):
            for j in range(n, 0, -1):
                dp[j] = max(dp[j], dp[j - 1] + nums1[i])
        
        for t in range(n + 1):
            if nums1_sum + total_increase * t - dp[t] <= x:
                return t
        
        return -1