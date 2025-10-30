from typing import List
from collections import Counter

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Count how many spells of each damage exist
        freq = Counter(power)

        # Get sorted unique damage values
        ds = sorted(freq)
        m = len(ds)

        # Weight of picking all spells of damage ds[i]
        W = [ds[i] * freq[ds[i]] for i in range(m)]

        # dp[i] = max total damage using damage values up to index i
        dp = [0] * m

        # Pointer j will track the last index whose damage is <= ds[i] - 3
        j = -1

        for i in range(m):
            # Advance j while the next damage is <= ds[i] - 3
            while j + 1 < i and ds[j + 1] <= ds[i] - 3:
                j += 1

            # If we include ds[i], we add W[i] plus dp[j] (or 0 if j < 0)
            include = W[i] + (dp[j] if j >= 0 else 0)

            # Either skip ds[i] (dp[i-1]) or include it
            dp[i] = include if i == 0 else max(dp[i - 1], include)

        # The answer is the best we can do using all damage values
        return dp[-1]