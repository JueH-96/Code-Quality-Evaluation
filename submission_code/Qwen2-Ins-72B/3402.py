from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        nums.sort()
        n = len(nums)
        MOD = 10**9 + 7
        
        def calculate_cost(target, cost1, cost2):
            cost = 0
            for num in nums:
                if num < target:
                    cost += (target - num) * cost1
                else:
                    diff = num - target
                    cost += min(diff * cost1, (diff // 2) * cost2 + ((diff % 2) * cost1))
            return cost % MOD
        
        median = nums[n // 2]
        cost = calculate_cost(median, cost1, cost2)
        
        return cost