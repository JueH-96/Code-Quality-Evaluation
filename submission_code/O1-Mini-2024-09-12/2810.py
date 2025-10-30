from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = nums.copy()
        total_sum = sum(min_cost)
        result = 0 * x + total_sum  # k=0
        
        for k in range(1, n):
            for j in range(n):
                shifted_index = (j - 1) % n
                min_cost[j] = min(min_cost[j], nums[shifted_index])
            current_sum = sum(min_cost)
            current_total = k * x + current_sum
            result = min(result, current_total) if k > 0 else current_total
            if k == 1:
                result = min(result, current_total)
            else:
                result = min(result, current_total)
        
        # To handle cases where k can be up to n-1
        min_total = float('inf')
        min_cost = [nums[j] for j in range(n)]
        for k in range(n):
            if k > 0:
                for j in range(n):
                    shifted_index = (j - 1) % n
                    min_cost[j] = min(min_cost[j], nums[shifted_index])
            current_sum = sum(min_cost)
            current_total = k * x + current_sum
            min_total = min(min_total, current_total)
        
        return min_total