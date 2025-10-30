from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # We can only increase, so optimal target T is the current maximum
        T = max(nums)
        # total increments needed
        S = sum(T - x for x in nums)
        # if pairing is not cheaper than two singles, do all singles
        if cost2 >= 2 * cost1:
            return (S * cost1) % MOD
        
        # otherwise, we use as many pair-operations as possible
        # largest single index requirement is from the smallest element
        d_max = T - min(nums)
        # the maximum number of pair-ops is limited by having two distinct indices:
        # at most floor(S/2), but also you cannot pair more than the sum of requirements
        # of the other indices, i.e. S - d_max
        max_pairs = min(S // 2, S - d_max)
        # compute cost
        cost = (max_pairs * cost2 + (S - 2 * max_pairs) * cost1) % MOD
        return cost