from typing import List
from math import gcd

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # If we have a coin of denomination 1, every positive integer can be formed
        if 1 in coins:
            return k
        
        n          = len(coins)
        min_coin   = min(coins)
        hi         = min_coin * k          # an upper-bound for the answer
        
        # ---------------------------------------------------------------------
        #  Pre–compute the lcm of every non–empty subset of coins.
        #  We also keep the sign (+/-) required by the inclusion–exclusion rule.
        # ---------------------------------------------------------------------
        def lcm(a: int, b: int) -> int:
            return a // gcd(a, b) * b
        
        lcm_cache  = [0] * (1 << n)        # lcm of each subset (bitmask index)
        subset_lcm = []                    # lcm values that do not exceed `hi`
        subset_sgn = []                    # corresponding +1 / –1
        
        for mask in range(1, 1 << n):
            # get index of a newly added coin
            lsb       = mask & -mask
            idx       = lsb.bit_length() - 1
            prev_mask = mask ^ lsb

            if prev_mask == 0:                           # single element subset
                cur_lcm = coins[idx]
            else:                                        # extend earlier subset
                prev_lcm = lcm_cache[prev_mask]
                if prev_lcm > hi:                        # already “infinite”
                    cur_lcm = hi + 1
                else:
                    cur_lcm = lcm(prev_lcm, coins[idx])
                    if cur_lcm > hi:
                        cur_lcm = hi + 1                 # treat as “infinite”

            lcm_cache[mask] = cur_lcm

            # store only useful subsets (lcm ≤ hi)
            if cur_lcm <= hi:
                subset_lcm.append(cur_lcm)
                subset_sgn.append(1 if (mask.bit_count() & 1) else -1)
        
        # ---------------------------------------------------------------
        #  Helper: how many representable amounts are ≤ x ?
        #  (Inclusion – Exclusion over all subsets.)
        # ---------------------------------------------------------------
        def count(x: int) -> int:
            total = 0
            for lc, sg in zip(subset_lcm, subset_sgn):
                if lc > x:
                    continue
                total += sg * (x // lc)
            return total
        
        # -------------------------
        #  Binary search on answer
        # -------------------------
        lo, hi_bound = 1, hi
        while lo < hi_bound:
            mid = (lo + hi_bound) // 2
            if count(mid) >= k:
                hi_bound = mid
            else:
                lo = mid + 1
        return lo