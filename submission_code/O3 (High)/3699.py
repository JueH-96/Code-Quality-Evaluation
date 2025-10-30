from typing import List
from collections import defaultdict
from math import gcd

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 7:                     # not enough places to build a special subsequence
            return 0

        ans = 0
        pairs_ratio_cnt = defaultdict(int)   # (numerator, denominator)  -> amount of (p,q) pairs

        # r is the index of the third element in the subsequence (original array index)
        # it must satisfy 2 <= r <= n-3   (because we need space for p,q before it and s after it)
        for r in range(2, n - 2):
            # The index q = r - 2 is now far enough from r (r - q > 1),
            # so every pair (p, q) with p <= q - 2 becomes legal for the current and all future r's.
            q = r - 2
            if q >= 2:                          # we need at least one index between p and q
                q_val = nums[q]
                # add all new pairs (p, q) to the global counter
                for p in range(q - 1):          # p runs from 0 .. q-2  (q - p > 1)
                    p_val = nums[p]
                    g = gcd(p_val, q_val)
                    ratio = (p_val // g, q_val // g)
                    pairs_ratio_cnt[ratio] += 1

            r_val = nums[r]

            # choose the last element s (s - r > 1  ->  s >= r + 2)
            for s in range(r + 2, n):
                s_val = nums[s]
                g = gcd(s_val, r_val)
                ratio = (s_val // g, r_val // g)

                # every stored (p,q) pair with the same ratio forms a valid quadruple
                ans += pairs_ratio_cnt.get(ratio, 0)

        return ans