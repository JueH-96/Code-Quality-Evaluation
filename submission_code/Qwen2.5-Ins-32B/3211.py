from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = 1
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        for i in range(1, n):
            dp[i] = 1
            low, high = 0, i
            while low < high:
                mid = (low + high) // 2
                if (prefix_sum[i + 1] - prefix_sum[mid + 1]) >= nums[mid] * (i - mid):
                    high = mid
                else:
                    low = mid + 1
            dp[i] = max(dp[i], dp[low - 1] + 1)
        
        return max(dp)