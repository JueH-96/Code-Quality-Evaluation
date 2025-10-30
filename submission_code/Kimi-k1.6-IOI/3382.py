from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute next greater elements for each index
        next_greater = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                j = stack.pop()
                next_greater[j] = i
            stack.append(i)
        
        # Create a dictionary to store sorted indices for each value
        sorted_indices = defaultdict(list)
        for idx, val in enumerate(nums):
            sorted_indices[val].append(idx)
        
        total = 0
        for i in range(n):
            x = nums[i]
            left_bound = i
            right_bound = next_greater[i] - 1
            
            # Get the list of indices for value x
            indices = sorted_indices[x]
            # Find the number of elements in [left_bound, right_bound]
            left = bisect_left(indices, left_bound)
            right = bisect_right(indices, right_bound)
            total += (right - left)
        
        return total