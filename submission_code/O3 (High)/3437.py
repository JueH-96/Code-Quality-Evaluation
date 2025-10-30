from typing import List
from collections import Counter
from bisect import bisect_right

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        """
        For every distinct damage value v we decide whether to take all spells
        that inflict v (worth v * cnt[v]) or to skip them.  
        Two taken damage values must differ by at least 3, because differences
        1 or 2 are forbidden.

        Sort the distinct values: val[0] < val[1] < … < val[m-1]

        DP:
            dp[i] – maximum total damage considering values up to val[i]

            when we are at i we have two options
                • skip val[i]      → dp[i-1]
                • take val[i]      → weight[i] + dp[p]
                                      where p is the last index with
                                      val[p] ≤ val[i] – 3   (-1 if none)

            dp[i] = max(skip, take)

        The answer is dp[-1].
        """
        cnt = Counter(power)
        if not cnt:
            return 0

        vals = sorted(cnt)                   # distinct damage values
        m = len(vals)
        dp = [0] * m

        for i, v in enumerate(vals):
            weight = v * cnt[v]              # total damage of all spells with value v

            # right-most index whose value is ≤ v-3
            idx = bisect_right(vals, v - 3) - 1
            take = weight + (dp[idx] if idx >= 0 else 0)
            skip = dp[i - 1] if i > 0 else 0

            dp[i] = max(skip, take)

        return dp[-1]