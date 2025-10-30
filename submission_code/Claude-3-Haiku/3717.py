from typing import List
from collections import defaultdict

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        
        # Create a dictionary to store the cumulative sum and its first index
        prefix_sum = {0: -1}
        curr_sum = 0
        for i, num in enumerate(nums):
            curr_sum += num
            prefix_sum[curr_sum] = i
        
        # Initialize the minimum number of operations
        min_ops = float('inf')
        
        # Iterate through the array and find the minimum number of operations
        curr_sum = 0
        for i in range(n):
            curr_sum += nums[i]
            target = curr_sum - (k * x)
            if target in prefix_sum:
                start = prefix_sum[target] + 1
                end = i
                if end - start + 1 == k:
                    min_ops = min(min_ops, i - start + 1)
        
        # Return the minimum number of operations or -1 if it's not possible
        return min_ops if min_ops != float('inf') else -1