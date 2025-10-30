from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total_cost = 0
        for i in range(n):
            total_cost += (nums[-1] - nums[i]) * cost1
        min_cost = total_cost
        for i in range(n - 1, 0, -1):
            diff = nums[i] - nums[i - 1]
            total_cost += (2 * diff * i - diff * n) * cost1
            total_cost -= diff * (n - i) * cost2
            min_cost = min(min_cost, total_cost)
        return min_cost % MOD