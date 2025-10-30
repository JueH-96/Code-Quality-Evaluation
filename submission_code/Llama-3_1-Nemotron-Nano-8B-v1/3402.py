from typing import List

MOD = 10**9 + 7

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        if not nums:
            return 0
        max_a = max(nums)
        min_a = min(nums)
        sum_a = sum(nums)
        n = len(nums)
        T = max_a
        sum_delta = n * T - sum_a
        if sum_delta == 0:
            return 0
        max_delta = T - min_a
        D = min(sum_delta // 2, sum_delta - max_delta)
        if cost2 < 2 * cost1:
            cost = (sum_delta - 2 * D) * cost1 + D * cost2
        else:
            cost = sum_delta * cost1
        return cost % MOD