from math import gcd
from collections import defaultdict
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_val = max(nums)
        count = defaultdict(int)
        for num in nums:
            for i in range(1, int(num**0.5) + 1):
                if num % i == 0:
                    count[i] += 1
                    if i * i != num:
                        count[num // i] += 1
        
        dp = [0] * (max_val + 1)
        pow2 = [1] * (len(nums) + 1)
        for i in range(1, len(nums) + 1):
            pow2[i] = (pow2[i - 1] * 2) % MOD
        
        for i in range(max_val, 0, -1):
            if count[i] == 0:
                continue
            dp[i] = pow2[count[i]] - 1
            for j in range(2 * i, max_val + 1, i):
                dp[i] = (dp[i] - dp[j] + MOD) % MOD
        
        ans = 0
        for i in range(1, max_val + 1):
            ans = (ans + dp[i] * (dp[i] - 1)) % MOD
        
        return ans