from typing import List
import bisect
import math

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # ------------------------------------------------------------------ #
        # 1.  Basic statistics on the input numbers                          #
        # ------------------------------------------------------------------ #
        max_val = max(nums)                     # maximum possible gcd value
        freq    = [0]*(max_val + 1)             # freq[x] = occurrences of x
        for x in nums:
            freq[x] += 1

        # ------------------------------------------------------------------ #
        # 2.  cntDiv[d] – how many numbers are divisible by d                #
        #     (classic harmonic–series loop, O(max_val log max_val))         #
        # ------------------------------------------------------------------ #
        cntDiv = [0]*(max_val + 1)
        for d in range(1, max_val + 1):
            s = 0
            for m in range(d, max_val + 1, d):      # all multiples of d
                s += freq[m]
            cntDiv[d] = s

        # ------------------------------------------------------------------ #
        # 3.  exact[g] – how many unordered pairs have gcd exactly g         #
        #     computed by inclusion–exclusion from big to small              #
        # ------------------------------------------------------------------ #
        exact = [0]*(max_val + 1)
        for g in range(max_val, 0, -1):
            c = cntDiv[g]
            pairs_divisible = c*(c-1)//2 if c >= 2 else 0      # C(c, 2)
            sub = 0
            for m in range(2*g, max_val + 1, g):               # multiples > g
                sub += exact[m]
            exact[g] = pairs_divisible - sub                   # remove over-counts

        # ------------------------------------------------------------------ #
        # 4.  Build prefix array of cumulative pair counts in ascending gcd  #
        # ------------------------------------------------------------------ #
        prefix, values = [], []        # prefix[i] – pairs up to values[i]
        cum = 0
        for g in range(1, max_val + 1):
            if exact[g]:
                cum += exact[g]
                prefix.append(cum)
                values.append(g)

        # ------------------------------------------------------------------ #
        # 5.  Answer queries with binary search                              #
        # ------------------------------------------------------------------ #
        ans = []
        for q in queries:
            # want first prefix > q, i.e. prefix >= q+1
            idx = bisect.bisect_left(prefix, q + 1)
            ans.append(values[idx])

        return ans