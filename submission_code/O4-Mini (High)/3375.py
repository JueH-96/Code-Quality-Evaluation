from typing import List
import math

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Minimum coin value
        c_min = min(coins)
        # Upper bound for the k-th smallest is k times the smallest coin
        high = c_min * k
        
        n = len(coins)
        total_masks = 1 << n
        
        # lcms[mask] = lcm of coins in the subset represented by mask
        # parity[mask] = 1 if the subset size is odd, 0 if even
        lcms = [0] * total_masks
        parity = [0] * total_masks
        
        # Build lcms and parity for every non-empty subset
        for mask in range(1, total_masks):
            # Extract one bit from mask
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            if prev == 0:
                # Subset of size 1
                lcms[mask] = coins[idx]
                parity[mask] = 1
            else:
                # Extend the previous subset
                prev_lcm = lcms[prev]
                c = coins[idx]
                g = math.gcd(prev_lcm, c)
                l = (prev_lcm // g) * c
                lcms[mask] = l
                # Flip parity
                parity[mask] = parity[prev] ^ 1
        
        # Count how many distinct amounts <= x are possible
        # via inclusion-exclusion over the arithmetic sequences
        def count_upto(x: int) -> int:
            total = 0
            for mask in range(1, total_masks):
                l = lcms[mask]
                if l > x:
                    continue
                cnt = x // l
                # If subset size odd, add; if even, subtract
                if parity[mask]:
                    total += cnt
                else:
                    total -= cnt
            return total
        
        # Binary search the k-th smallest amount
        lo, hi = 1, high
        while lo < hi:
            mid = (lo + hi) // 2
            if count_upto(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        
        return lo