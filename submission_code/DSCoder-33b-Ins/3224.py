from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = (dp[i - 1] + dp[i - 2]) % MOD
        ans = 1
        sick.append(n)
        prev = 0
        for i in sick:
            ans = ans * dp[i - prev] % MOD
            prev = i + 1
        return ans