from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        res = 0
        
        for pos, x in queries:
            nums[pos] = x
            n = len(nums)
            dp = [0] * n
            dp[0] = max(0, nums[0])
            if n > 1:
                dp[1] = max(dp[0], nums[1])
            for i in range(2, n):
                dp[i] = max(dp[i-1], dp[i-2] + nums[i])
            res = (res + dp[-1]) % MOD
        
        return res