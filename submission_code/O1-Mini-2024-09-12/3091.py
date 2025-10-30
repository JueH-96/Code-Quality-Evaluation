from typing import List
from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        count = Counter(nums)
        dp = [0] * (r + 1)
        dp[0] = 1
        for num, freq in count.items():
            for s in range(r, num - 1, -1):
                for k in range(1, freq + 1):
                    if s - k * num < 0:
                        break
                    dp[s] = (dp[s] + dp[s - k * num]) % MOD
        total = sum(dp[l:r+1]) % MOD
        return total