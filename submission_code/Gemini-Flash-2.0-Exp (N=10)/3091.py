from collections import Counter
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        counts = Counter(nums)
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1

        for num, count in counts.items():
            for i in range(max_sum, -1, -1):
                for j in range(1, count + 1):
                    if i - num * j >= 0:
                        dp[i] = (dp[i] + dp[i - num * j]) % MOD
                    else:
                        break

        result = 0
        for i in range(l, r + 1):
            result = (result + dp[i]) % MOD
        
        return result