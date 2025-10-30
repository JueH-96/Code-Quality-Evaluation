from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total_cost = 0
        
        # Calculate the median
        median = nums[n // 2]
        
        # Calculate the cost to make all elements equal to the median
        for num in nums:
            diff = abs(num - median)
            pairs = diff // 2
            singles = diff % 2
            
            total_cost += (pairs * cost2) + (singles * cost1)
            total_cost %= MOD
        
        return total_cost