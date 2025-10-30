import math
from typing import List

MOD = 10**9 + 7

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        max_val = max(nums)
        n = len(nums)
        ans = 0
        for g in range(1, max_val + 1):
            Q = []
            for x in nums:
                if x % g == 0:
                    Q.append(x // g)
            if not Q:
                continue
            M = max(Q)
            dp = [[0] * (M + 1) for _ in range(M + 1)]
            dp[0][0] = 1
            for x in Q:
                new_dp = [[0] * (M + 1) for _ in range(M + 1)]
                for a in range(M + 1):
                    for b in range(M + 1):
                        if dp[a][b] == 0:
                            continue
                        new_dp[a][b] = (new_dp[a][b] + dp[a][b]) % MOD
                        a_new = math.gcd(a, x)
                        new_dp[a_new][b] = (new_dp[a_new][b] + dp[a][b]) % MOD
                        b_new = math.gcd(b, x)
                        new_dp[a][b_new] = (new_dp[a][b_new] + dp[a][b]) % MOD
                dp = new_dp
            ans = (ans + dp[1][1]) % MOD
        return ans