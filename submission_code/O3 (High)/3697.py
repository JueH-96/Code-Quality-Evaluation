from typing import List
from math import gcd


class Solution:
    # helper to compute lcm of two integers
    def _lcm(self, a: int, b: int) -> int:
        return a // gcd(a, b) * b

    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        k = len(target)                       # 1 ≤ k ≤ 4
        full_mask = 1 << k                    # 2 ** k
        INF = 10 ** 20                        # big enough

        # ------------------------------------------------------------------
        # Pre-compute LCM for every non–empty subset of the target array
        # mask : bit i is 1 if target[i] is in the subset
        # ------------------------------------------------------------------
        subset_lcm = [1] * full_mask          # subset_lcm[0] is dummy (unused)

        for mask in range(1, full_mask):
            # take out the lowest set bit to build lcm incrementally
            bit = (mask & -mask).bit_length() - 1       # position of least-significant 1
            prev = mask & (mask - 1)                    # mask without that bit
            if prev == 0:
                subset_lcm[mask] = target[bit]
            else:
                subset_lcm[mask] = self._lcm(subset_lcm[prev], target[bit])

        # ------------------------------------------------------------------
        # Dynamic programming over the 2^k possible “already covered” masks
        # dp[mask] – minimal total increments needed so far to cover `mask`
        # ------------------------------------------------------------------
        dp = [INF] * full_mask
        dp[0] = 0                                       # nothing covered, zero cost

        for num in nums:
            # cost_to_cover[mask]: cost of turning `num` into a multiple of lcm(mask)
            cost_to_cover = [0] * full_mask
            for mask in range(1, full_mask):
                l = subset_lcm[mask]
                cost_to_cover[mask] = (l - (num % l)) % l    # 0 … l-1

            # update dp using this particular number **at most once**
            new_dp = dp[:]                                # copy current bests
            for mask in range(1, full_mask):              # subsets we let this number cover
                cost = cost_to_cover[mask]
                for prev_mask in range(full_mask):
                    if dp[prev_mask] == INF:
                        continue
                    merged = prev_mask | mask
                    if new_dp[merged] > dp[prev_mask] + cost:
                        new_dp[merged] = dp[prev_mask] + cost
            dp = new_dp                                    # move to next number

        return dp[full_mask - 1]                           # all targets covered