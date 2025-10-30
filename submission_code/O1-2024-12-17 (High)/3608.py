import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_val = 200

        # Precompute gcd for all pairs up to 200 (including gcd with 0 to represent empty-subset gcd).
        gcd_table = [[0]*(max_val+1) for _ in range(max_val+1)]
        for x in range(max_val+1):
            gcd_table[x][0] = x
            gcd_table[0][x] = x
        for x in range(1, max_val+1):
            for y in range(1, max_val+1):
                gcd_table[x][y] = math.gcd(x, y)

        # dp[g1][g2] = number of ways (so far) to form two subsequences (seq1, seq2)
        # such that gcd(seq1) = g1, gcd(seq2) = g2, using some prefix of nums.
        # We start with dp[0][0] = 1 (both subsequences empty).
        dp = [[0]*(max_val+1) for _ in range(max_val+1)]
        dp[0][0] = 1

        # Process each element: we can skip it, put it in seq1, or put it in seq2.
        for x in nums:
            next_dp = [[0]*(max_val+1) for _ in range(max_val+1)]
            for g1 in range(max_val+1):
                for g2 in range(max_val+1):
                    ways = dp[g1][g2]
                    if ways == 0:
                        continue

                    # Option 1: skip x (do not include in either subsequence)
                    next_dp[g1][g2] = (next_dp[g1][g2] + ways) % MOD

                    # Option 2: put x in seq1 => new gcd is gcd_table[g1][x]
                    new_g1 = gcd_table[g1][x]
                    next_dp[new_g1][g2] = (next_dp[new_g1][g2] + ways) % MOD

                    # Option 3: put x in seq2 => new gcd is gcd_table[g2][x]
                    new_g2 = gcd_table[g2][x]
                    next_dp[g1][new_g2] = (next_dp[g1][new_g2] + ways) % MOD

            dp = next_dp

        # We want both subsequences to be non-empty and to have the same gcd.
        # "Non-empty" means gcd != 0 in our DP. So we sum dp[g][g] for g=1..200.
        ans = 0
        for g in range(1, max_val+1):
            ans = (ans + dp[g][g]) % MOD

        return ans