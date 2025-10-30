from typing import List
from math import gcd

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # We'll use binary search + inclusion-exclusion to count how many
        # distinct multiples are <= x. We then find the smallest x s.t.
        # count(x) >= k, which must be the k-th smallest.

        n = len(coins)
        
        # Precompute LCMs for all nonempty subsets of coins
        # lcms[mask] = LCM of the subset indicated by 'mask'
        # mask goes from 1..(1<<n)-1, each bit in mask indicates choosing a coin
        lcms = [0] * (1 << n)
        
        # lcm utility
        def lcm(a, b):
            return a // gcd(a, b) * b
        
        coins_sorted = sorted(coins)
        # Build up LCM table using dynamic programming over subsets
        for mask in range(1, 1 << n):
            # isolate the lowest set bit
            lowest_bit = mask & -mask
            j = (lowest_bit).bit_length() - 1  # index of that bit
            pmask = mask ^ lowest_bit
            if pmask == 0:
                # subset of size 1
                lcms[mask] = coins_sorted[j]
            else:
                lcms[mask] = lcm(lcms[pmask], coins_sorted[j])
        
        # count(x) = number of distinct multiples <= x
        # by inclusion-exclusion: add floor(x/lcm(subset)) for
        # odd-size subsets, subtract for even-size subsets
        def count_up_to(x):
            total = 0
            for mask in range(1, 1 << n):
                # parity of subset size
                sz = mask.bit_count()  # count bits
                sign = 1 if (sz % 2) == 1 else -1
                # if lcms[mask] > x, floor(x/lcms[mask]) = 0, so it adds nothing
                total += sign * (x // lcms[mask])
            return total
        
        # Binary search to find the k-th smallest
        # The largest coin is at most 25, hence the k-th multiple can be at most 25*k
        left, right = 1, 25 * k
        while left < right:
            mid = (left + right) // 2
            if count_up_to(mid) >= k:
                right = mid
            else:
                left = mid + 1
        
        return left