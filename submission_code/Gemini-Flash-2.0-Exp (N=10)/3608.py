import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_val = max(nums)
        counts = [0] * (max_val + 1)
        for num in nums:
            counts[num] += 1
        
        dp = [0] * (max_val + 1)
        for i in range(1, max_val + 1):
            c = 0
            for j in range(i, max_val + 1, i):
                c += counts[j]
            if c > 0:
                dp[i] = pow(2, c, MOD) - 1
        
        for i in range(max_val, 0, -1):
            for j in range(2 * i, max_val + 1, i):
                dp[i] = (dp[i] - dp[j] + MOD) % MOD
        
        ans = 0
        for i in range(1, max_val + 1):
            ans = (ans + dp[i] * dp[i]) % MOD
        
        return ans