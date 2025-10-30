from typing import List
import math

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_val = max(nums)
        count = [0] * (max_val + 1)
        for num in nums:
            count[num] += 1
        
        dp = [0] * (max_val + 1)
        for i in range(1, max_val + 1):
            for j in range(i, max_val + 1, i):
                dp[i] = (dp[i] + count[j]) % MOD
            dp[i] = pow(2, dp[i], MOD) - 1
        
        ans = 0
        for i in range(max_val, 0, -1):
            curr = dp[i]
            for j in range(2 * i, max_val + 1, i):
                curr = (curr - dp[j] + MOD) % MOD
            ans = (ans + curr * curr) % MOD
        
        return ans