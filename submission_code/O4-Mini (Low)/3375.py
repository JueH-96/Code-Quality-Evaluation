from typing import List
import math

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)
        # Precompute for all non-empty subsets:
        # their lcm, the parity of subset size (for inclusion-exclusion sign)
        subset_lcms = []
        for mask in range(1, 1 << n):
            l = 1
            bits = 0
            # compute lcm of coins in this subset
            m = mask
            i = 0
            while m:
                if m & 1:
                    bits += 1
                    l = l * coins[i] // math.gcd(l, coins[i])
                i += 1
                m >>= 1
                # early break if l is already huge
            # inclusion-exclusion sign
            sign = 1 if (bits % 2) == 1 else -1
            subset_lcms.append((l, sign))
        
        # function to count how many distinct multiples <= x
        # using inclusion-exclusion over the subsets
        def count_upto(x: int) -> int:
            total = 0
            for l, sign in subset_lcms:
                total += sign * (x // l)
            return total
        
        # binary search on the answer
        low, high = 1, k * min(coins)
        while low < high:
            mid = (low + high) // 2
            if count_upto(mid) >= k:
                high = mid
            else:
                low = mid + 1
        return low