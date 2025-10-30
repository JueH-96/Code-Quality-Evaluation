from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum1 = sum(nums1)
        total_sum2 = sum(nums2)
        
        if total_sum1 <= x:
            return 0
        
        pairs = sorted(zip(nums2, nums1), reverse=True)
        nums2, nums1 = zip(*pairs)
        
        dp = [0] * (n + 1)
        
        for t in range(1, n + 1):
            for i in range(t, -1, -1):
                dp[i] = max(dp[i], dp[i - 1] + nums2[t - 1] * i + nums1[t - 1])
                if total_sum1 + total_sum2 * t - dp[i] <= x:
                    return t
        
        return -1