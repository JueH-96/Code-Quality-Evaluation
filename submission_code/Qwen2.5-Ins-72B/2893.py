from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        
        # Initialize the last seen positions for even and odd parities
        last_even = 0 if nums[0] % 2 == 0 else -1
        last_odd = 0 if nums[0] % 2 != 0 else -1
        
        for i in range(1, n):
            if nums[i] % 2 == 0:
                if last_even != -1:
                    dp[i] = max(dp[i], dp[last_even] + nums[i])
                if last_odd != -1:
                    dp[i] = max(dp[i], dp[last_odd] + nums[i] - x)
                last_even = i
            else:
                if last_odd != -1:
                    dp[i] = max(dp[i], dp[last_odd] + nums[i])
                if last_even != -1:
                    dp[i] = max(dp[i], dp[last_even] + nums[i] - x)
                last_odd = i
        
        return max(dp)