from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_total_cost = float('inf')
        
        for max_k in range(n):
            operation_cost = max_k * x
            collection_cost = 0
            for s in range(n):
                min_cost = float('inf')
                for k in range(max_k + 1):
                    current_type = (s - k) % n
                    min_cost = min(min_cost, nums[current_type])
                collection_cost += min_cost
            total_cost = operation_cost + collection_cost
            min_total_cost = min(min_total_cost, total_cost)
        
        return min_total_cost