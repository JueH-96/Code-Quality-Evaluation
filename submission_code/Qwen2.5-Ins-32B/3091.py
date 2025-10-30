from collections import Counter
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        counter = Counter(nums)
        dp = [0] * (r + 1)
        dp[0] = 1  # The empty subset
        
        for num, count in counter.items():
            for i in range(num, r + 1):
                for j in range(1, count + 1):
                    if i - num * j >= 0:
                        dp[i] = (dp[i] + dp[i - num * j]) % MOD
        
        return sum(dp[l:r + 1]) % MOD