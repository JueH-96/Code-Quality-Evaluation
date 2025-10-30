import math
from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Initialize min_total_cost to a very large number.
        min_total_cost = math.inf
        
        # The first subarray always starts at index 0, so its cost is nums[0].
        # We need to find the optimal split points for the second and third subarrays.
        # Let the three subarrays be:
        # 1. nums[0 : i]    (starts at nums[0])
        # 2. nums[i : j]    (starts at nums[i])
        # 3. nums[j : n]    (starts at nums[j])
        
        # Constraints for i and j to ensure all three subarrays are non-empty:
        # - Subarray 1: nums[0 ... i-1] must have at least 1 element, so i >= 1.
        # - Subarray 2: nums[i ... j-1] must have at least 1 element, so j >= i + 1.
        # - Subarray 3: nums[j ... n-1] must have at least 1 element, so j <= n - 1.
        
        # The loop for 'i' (start of the second subarray):
        # 'i' can start from 1. The largest 'i' can be is n-2, because
        # if i = n-2, then nums[n-2] is the start of the second subarray.
        # This leaves nums[n-1] for the third subarray (j=n-1).
        # This ensures there's at least one element for the second and third subarrays after index i-1.
        for i in range(1, n - 1):
            # The loop for 'j' (start of the third subarray):
            # 'j' must be at least i + 1 (to ensure the second subarray is non-empty).
            # 'j' can go up to n - 1 (as nums[n-1] is the last possible starting element for the third subarray).
            for j in range(i + 1, n):
                # Calculate the total cost for the current partition
                # Cost = cost_of_subarray1 + cost_of_subarray2 + cost_of_subarray3
                current_cost = nums[0] + nums[i] + nums[j]
                
                # Update the minimum total cost found so far
                min_total_cost = min(min_total_cost, current_cost)
                
        return min_total_cost