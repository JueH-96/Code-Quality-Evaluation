from math import factorial
from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1
        for end, cnt in reversed(requirements):
            new_dp = [0] * (n + 1)
            for i in range(end + 1):
                for j in range(cnt, min(i, n) + 1):
                    new_dp[j] += dp[j - cnt] * factorial(i) // (factorial(cnt) * factorial(i - cnt))
                    new_dp[j] %= MOD
            dp = new_dp
        return dp[n]