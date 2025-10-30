from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        max_val = max(nums)
        min_cost = float('inf')
        
        for target in range(max_val, max_val + len(nums)):
            cost = 0
            count = 0
            for num in nums:
                diff = target - num
                if diff % 2 == 0:
                    cost += diff // 2 * cost2
                else:
                    cost += (diff // 2) * cost2 + cost1
                    count += 1
            min_cost = min(min_cost, cost)
        
        return min_cost % MOD