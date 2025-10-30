import math
from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Initialize min_total_cost to positive infinity.
        # Max element value is 50, so max sum of 3 costs is 150.
        # math.inf is a robust choice.
        min_total_cost = math.inf
        
        # The cost of the first subarray is fixed as nums[0], because the
        # first subarray must start at index 0 and be non-empty.
        cost1 = nums[0]
        
        # k1 is the starting index of the second subarray.
        # It ranges from 1 to n-2.
        # range(1, n-1) in Python means indices 1, 2, ..., n-2.
        for k1 in range(1, n - 1):
            cost2 = nums[k1]
            
            # k2 is the starting index of the third subarray.
            # It ranges from k1 + 1 to n-1.
            # range(k1 + 1, n) in Python means indices k1 + 1, ..., n-1.
            for k2 in range(k1 + 1, n):
                cost3 = nums[k2]
                
                current_total_cost = cost1 + cost2 + cost3
                if current_total_cost < min_total_cost:
                    min_total_cost = current_total_cost
                    
        return min_total_cost