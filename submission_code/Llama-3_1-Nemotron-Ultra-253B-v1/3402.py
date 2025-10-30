from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        if not nums:
            return 0
        target = max(nums)
        increments = [target - num for num in nums]
        S = sum(increments)
        max_inc = max(increments) if increments else 0
        
        if cost2 < 2 * cost1:
            pair_ops = min(S // 2, S - max_inc)
            remaining = S - 2 * pair_ops
            cost = (pair_ops * cost2 + remaining * cost1) % (10**9 + 7)
        else:
            cost = (S * cost1) % (10**9 + 7)
        
        return cost