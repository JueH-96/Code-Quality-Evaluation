from typing import List
from collections import defaultdict

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        def cost_to_make_equal(subarray):
            # Find the median of the subarray
            subarray.sort()
            median = subarray[len(subarray) // 2]
            # Calculate the cost to make all elements equal to the median
            return sum(abs(num - median) for num in subarray)
        
        n = len(nums)
        min_operations = float('inf')
        
        # Try every possible starting point for k subarrays of size x
        for start in range(n - k * x + 1):
            total_cost = 0
            for i in range(k):
                subarray = nums[start + i * x : start + (i + 1) * x]
                total_cost += cost_to_make_equal(subarray)
            min_operations = min(min_operations, total_cost)
        
        return min_operations