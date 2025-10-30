from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_nums = nums.copy()
        for i in range(1, n):
            for j in range(n):
                min_nums[j] = min(min_nums[j], nums[(j + i) % n])
        
        min_cost = [float('inf')] * n
        for i in range(n):
            min_cost[i] = min(min_cost[i], nums[i])
            for j in range(n):
                min_cost[i] = min(min_cost[i], min_nums[j] + x * j)
        
        return min(min_cost)