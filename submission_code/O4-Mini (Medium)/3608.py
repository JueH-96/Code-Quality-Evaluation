from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        maxv = max(nums)
        ans = 0
        # For each g, collect the values a_i = nums[i] // g for nums[i] divisible by g
        for g in range(1, maxv + 1):
            # build list of a_i
            vals = []
            for x in nums:
                if x % g == 0:
                    vals.append(x // g)
            M = len(vals)
            # we need at least two elements to form two non-empty disjoint subsequences
            if M < 2:
                continue
            # dp[(d1, d2)] = count of ways so far to assign some prefix of vals
            # into s1 with current gcd d1 (0 means empty), and s2 with gcd d2 (0 means empty)
            dp = defaultdict(int)
            dp[(0, 0)] = 1
            for a in vals:
                # snapshot old dp
                old_items = list(dp.items())
                for (d1, d2), cnt in old_items:
                    # assign a to s1
                    nd1 = a if d1 == 0 else gcd(d1, a)
                    key1 = (nd1, d2)
                    dp[key1] = (dp[key1] + cnt) % mod
                    # assign a to s2
                    nd2 = a if d2 == 0 else gcd(d2, a)
                    key2 = (d1, nd2)
                    dp[key2] = (dp[key2] + cnt) % mod
                # skipping a is already in dp via old states
            # result for this g is dp[(1,1)] (both non-empty and gcd==1 after division)
            ans = (ans + dp.get((1, 1), 0)) % mod
        return ans