import collections
import bisect
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        cnt = collections.Counter(power)
        dam_list = sorted(cnt.keys())
        m = len(dam_list)
        if m == 0:
            return 0
        weight_list = [cnt[dam] * dam for dam in dam_list]
        dp = [0] * m
        dp[0] = max(0, weight_list[0])
        for i in range(1, m):
            target = dam_list[i] - 3
            idx = bisect.bisect_right(dam_list, target)
            if idx > 0:
                j = idx - 1
                val = dp[j]
            else:
                val = 0
            take_val = weight_list[i] + val
            not_take_val = dp[i - 1]
            dp[i] = max(not_take_val, take_val)
        return dp[m - 1]