import collections
from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # Precompute prefix ORs
        # left_ors[i] will store the bitwise OR of nums[0] through nums[i-1].
        # For i=0, left_ors[0] will be 0 (OR of an empty set).
        left_ors = [0] * n
        current_prefix_or = 0
        for i in range(n):
            left_ors[i] = current_prefix_or
            current_prefix_or |= nums[i]

        # Precompute suffix ORs
        # right_ors[i] will store the bitwise OR of nums[i+1] through nums[n-1].
        # For i=n-1, right_ors[n-1] will be 0 (OR of an empty set).
        right_ors = [0] * n
        current_suffix_or = 0
        for i in range(n - 1, -1, -1):
            right_ors[i] = current_suffix_or
            current_suffix_or |= nums[i]
        
        max_total_or = 0

        # Iterate through each number in nums
        # For each nums[i], consider the scenario where it receives all k operations
        # and all other numbers remain unchanged.
        for i in range(n):
            # Calculate the value of nums[i] after applying k operations.
            # This is equivalent to multiplying by 2^k, or left-shifting by k bits.
            transformed_num_i = nums[i] << k
            
            # Calculate the total bitwise OR for this configuration:
            # (OR of elements before nums[i]) | (transformed nums[i]) | (OR of elements after nums[i])
            current_total_or = left_ors[i] | transformed_num_i | right_ors[i]
            
            # Update the maximum OR found so far
            if current_total_or > max_total_or:
                max_total_or = current_total_or
        
        return max_total_or