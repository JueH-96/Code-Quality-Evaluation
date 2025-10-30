from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0] * (k + 1)
        dp[0] = 1
        
        for num in nums:
            for j in range(k, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % MOD
        
        result = pow(2, n, MOD) * dp[k] % MOD
        return result