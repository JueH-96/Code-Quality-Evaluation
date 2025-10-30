from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Compute the maximum value among nums[0:j] for each j.
        # prefix_max[j] will hold max(nums[0], nums[1], ..., nums[j-1])
        prefix_max = [0] * n
        prefix_max[0] = nums[0]
        for j in range(1, n):
            prefix_max[j] = max(prefix_max[j-1], nums[j-1])
        
        # Compute the maximum value among nums[j+1:n] for each j.
        # suffix_max[j] will hold max(nums[j+1], ..., nums[n-1])
        suffix_max = [0] * n
        suffix_max[n-1] = nums[n-1]
        for j in range(n-2, -1, -1):
            suffix_max[j] = max(suffix_max[j+1], nums[j+1])
        
        max_val = float("-inf")
        # Iterate through each possible middle index j (with i<j<k) and compute the triplet value.
        for j in range(1, n-1):
            candidate = (prefix_max[j] - nums[j]) * suffix_max[j]
            max_val = max(max_val, candidate)
            
        # If the best candidate is negative, we return 0 per the problem description.
        return max(0, max_val)