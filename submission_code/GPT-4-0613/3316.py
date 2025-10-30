from typing import List
from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        dp = [[0]*(k+1) for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            dp[i][0] = 1
            for j in range(1, min(i, k)+1):
                dp[i][j] = (dp[i-1][j-1] + dp[i-1][j]*2) % MOD
        res = 0
        for i in range(n):
            res = (res + (dp[i][k-1] - dp[i-1][k-1]) * nums[i]) % MOD
            res = (res - (dp[n-i-1][k-1] - dp[n-i][k-1]) * nums[i]) % MOD
        return res