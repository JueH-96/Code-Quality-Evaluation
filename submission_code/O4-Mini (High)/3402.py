from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        N = len(nums)
        total = sum(nums)
        # If only one element, no cost needed.
        if N <= 1:
            return 0
        # Maximum and minimum in the array
        mx = max(nums)
        mn = min(nums)
        # If doubling is not cheaper than two singles, never use it.
        if cost2 >= 2 * cost1:
            # Just pay single increments to raise all to mx
            need = N * mx - total
            return (need * cost1) % MOD

        # Otherwise doubling can help. We will consider a few candidate targets.
        # Always consider target = current maximum.
        candidates = {mx}

        # For N >= 3, the cost‐vs‐double‐capacity function changes slope at
        # T0 = (sum(nums) - 2*min(nums)) / (N-2).  We try ceil(T0) and one below.
        if N >= 3:
            B = total
            num = B - 2 * mn
            den = N - 2
            T1 = (num + den - 1) // den  # ceil of (B - 2*mn)/(N-2)
            if T1 >= mx:
                candidates.add(T1)
            if T1 - 1 >= mx:
                candidates.add(T1 - 1)

        ans = float('inf')
        for T in candidates:
            # Total increments needed
            S = N * T - total
            # Maximum increments needed by one element
            max_delta = T - mn
            # Max double ops = min( floor(S/2), S - max_delta )
            dbl_cap1 = S // 2
            dbl_cap2 = S - max_delta
            d = dbl_cap1 if dbl_cap1 < dbl_cap2 else dbl_cap2
            if d < 0:
                d = 0
            # Remaining singles
            s = S - 2 * d
            cost = (d * cost2 + s * cost1) % MOD
            if cost < ans:
                ans = cost

        return ans