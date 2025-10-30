from typing import List
import math

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # If there's only one denomination, the answer is just that denomination times k.
        if len(coins) == 1:
            return coins[0] * k

        m = len(coins)
        # Precompute lcm for every non-empty subset of coins, and the sign for inclusion‐exclusion.
        # We'll index subsets by a bitmask 1..(1<<m)-1.
        nsub = 1 << m
        lcms = [0] * nsub
        parity = [0] * nsub  # parity[mask] = number of bits in mask mod 2
        lcms[0] = 1
        parity[0] = 0

        for mask in range(1, nsub):
            # find lowest set bit
            lb = mask & -mask
            prev = mask ^ lb
            idx = (lb.bit_length() - 1)  # which coin we added
            # compute lcm of subset = lcm( lcm(prev_subset), coins[idx] )
            prev_lcm = lcms[prev]
            c = coins[idx]
            l = prev_lcm // math.gcd(prev_lcm, c) * c
            lcms[mask] = l
            parity[mask] = parity[prev] ^ 1

        # Build list of (lcm, sign) for all non-empty subsets
        subs = []
        for mask in range(1, nsub):
            sign = 1 if parity[mask] == 1 else -1
            subs.append((lcms[mask], sign))

        # Given X, count how many distinct amounts ≤ X can be formed
        # by the union of multiples of each coin.
        def count_upto(X: int) -> int:
            total = 0
            for l, s in subs:
                total += s * (X // l)
            return total

        # Binary search for the smallest X such that count_upto(X) >= k
        lo = 1
        hi = k * min(coins)  # the k-th smallest can't exceed this
        while lo < hi:
            mid = (lo + hi) // 2
            if count_upto(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo