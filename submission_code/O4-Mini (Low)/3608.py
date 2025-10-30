from typing import List
import math

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        maxv = max(nums)
        # dp[g1][g2]: number of ways so far to choose seq1 with gcd g1 (0 means empty),
        # and seq2 with gcd g2 (0 means empty)
        # initialize
        dp = [[0] * (maxv + 1) for _ in range(maxv + 1)]
        dp[0][0] = 1
        
        for a in nums:
            newdp = [[0] * (maxv + 1) for _ in range(maxv + 1)]
            for g1 in range(maxv + 1):
                for g2 in range(maxv + 1):
                    ways = dp[g1][g2]
                    if ways == 0:
                        continue
                    # 1) skip a
                    newdp[g1][g2] = (newdp[g1][g2] + ways) % MOD
                    # 2) put in seq1
                    ng1 = a if g1 == 0 else math.gcd(g1, a)
                    newdp[ng1][g2] = (newdp[ng1][g2] + ways) % MOD
                    # 3) put in seq2
                    ng2 = a if g2 == 0 else math.gcd(g2, a)
                    newdp[g1][ng2] = (newdp[g1][ng2] + ways) % MOD
            dp = newdp
        
        # sum over g>0 where both gcds equal
        ans = 0
        for g in range(1, maxv + 1):
            ans = (ans + dp[g][g]) % MOD
        return ans