import math
from typing import List

class Solution:
    def _is_non_decreasing(self, arr: List[int]) -> bool:
        """Helper function to check if an array is non-decreasing.
        An array with 0 or 1 element is always non-decreasing.
        """
        if len(arr) <= 1:
            return True
        for i in range(len(arr) - 1):
            if arr[i] > arr[i+1]:
                return False
        return True

    def minimumPairRemoval(self, nums: List[int]) -> int:
        current_nums = list(nums)  # Create a mutable copy to avoid modifying the input list
        operations = 0

        # Loop until the array becomes non-decreasing
        while True:
            # 1. Check if the array is already non-decreasing
            if self._is_non_decreasing(current_nums):
                return operations

            # 2. Find the adjacent pair with the minimum sum
            min_sum = float('inf')
            min_idx = -1
            
            # The array must have at least 2 elements to form an adjacent pair
            # when this part of the loop is reached (otherwise _is_non_decreasing
            # would have returned True for lengths 0 or 1).
            for i in range(len(current_nums) - 1):
                current_pair_sum = current_nums[i] + current_nums[i+1]
                
                # "If multiple such pairs exist, choose the leftmost one."
                # By using '<' and iterating from left to right, if a sum is equal to min_sum,
                # we don't update min_idx, naturally preserving the leftmost one found so far.
                if current_pair_sum < min_sum:
                    min_sum = current_pair_sum
                    min_idx = i
            
            # At this point, min_idx must have been updated to a valid index
            # because len(current_nums) >= 2 guarantees at least one pair exists.

            # 3. Replace the selected pair with their sum
            new_val = current_nums[min_idx] + current_nums[min_idx+1]
            
            # Construct the new array.
            # Example: current_nums = [A, B, C, D, E], min_idx = 2 (for C,D)
            # becomes [A, B, (C+D), E]
            current_nums = current_nums[:min_idx] + [new_val] + current_nums[min_idx+2:]
            
            # 4. Increment operation count
            operations += 1