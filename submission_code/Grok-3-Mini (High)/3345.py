import math
from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 1000000007
        n = len(nums)
        # dp[i][s][m] : number of subsets using first i elements with sum s and size m
        dp = [[[0 for _ in range(n + 1)] for _ in range(k + 1)] for _ in range(n + 1)]
        dp[0][0][0] = 1
        
        for i in range(1, n + 1):
            num = nums[i - 1]
            for s in range(k + 1):
                for m in range(n + 1):
                    # Not taking the i-th element
                    dp[i][s][m] = dp[i - 1][s][m]
                    # Taking the i-th element, if possible
                    if s >= num and m >= 1:
                        dp[i][s][m] += dp[i - 1][s - num][m - 1]
                        dp[i][s][m] %= MOD
        
        # Now sum over all m for sum k
        ans = 0
        for m in range(0, n + 1):
            # dp[n][k][m] * (2^(n - m)) mod MOD
            prod = (dp[n][k][m] * pow(2, n - m, MOD)) % MOD
            ans += prod
            ans %= MOD
        
        return ans