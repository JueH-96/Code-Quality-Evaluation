from typing import List
import math

MOD = 10 ** 9 + 7
MAX_VAL = 200            # 1 <= nums[i] <= 200


class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        # ------------------------------------------------------------------
        # pre–compute gcd for every pair of values in [0 .. 200]
        # gcd(0,x) is understood as x (0 means “empty subsequence” here).
        # ------------------------------------------------------------------
        gcd_tab = [[0] * (MAX_VAL + 1) for _ in range(MAX_VAL + 1)]
        for a in range(MAX_VAL + 1):
            for b in range(MAX_VAL + 1):
                if a == 0:
                    gcd_tab[a][b] = b
                elif b == 0:
                    gcd_tab[a][b] = a
                else:
                    gcd_tab[a][b] = math.gcd(a, b)

        # ------------------------------------------------------------------
        # dp[g1][g2]  – number of ways to process the current prefix of nums
        #               so that
        #                 • gcd of the first subsequence is g1 (0 ⇒ empty)
        #                 • gcd of the second subsequence is g2 (0 ⇒ empty)
        #
        # start with no element chosen: gcds are (0, 0)
        # ------------------------------------------------------------------
        dp = [[0] * (MAX_VAL + 1) for _ in range(MAX_VAL + 1)]
        dp[0][0] = 1

        for num in nums:
            nxt = [[0] * (MAX_VAL + 1) for _ in range(MAX_VAL + 1)]

            for g1 in range(MAX_VAL + 1):
                row = dp[g1]
                for g2 in range(MAX_VAL + 1):
                    cnt = row[g2]
                    if cnt == 0:
                        continue

                    # 1) leave num unused
                    nxt[g1][g2] = (nxt[g1][g2] + cnt) % MOD

                    # 2) put num into the first subsequence
                    ng1 = gcd_tab[g1][num]
                    nxt[ng1][g2] = (nxt[ng1][g2] + cnt) % MOD

                    # 3) put num into the second subsequence
                    ng2 = gcd_tab[g2][num]
                    nxt[g1][ng2] = (nxt[g1][ng2] + cnt) % MOD

            dp = nxt  # move to the next position

        # ------------------------------------------------------------------
        # answer: both subsequences have to be non-empty and share the same
        # gcd – i.e. states (g, g) with g ≥ 1.
        # ------------------------------------------------------------------
        ans = 0
        for g in range(1, MAX_VAL + 1):
            ans = (ans + dp[g][g]) % MOD
        return ans