from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        
        # Helper function to check if the array is non-decreasing
        # An array is non-decreasing if each element is greater than or equal to the previous one.
        # Arrays of length 0 or 1 are considered non-decreasing.
        def is_non_decreasing(arr):
            # The loop range(len(arr) - 1) handles arrays of length 0 or 1 correctly
            # by producing an empty range, causing the loop to be skipped and True returned.
            for i in range(len(arr) - 1):
                if arr[i] > arr[i+1]:
                    return False
            return True

        operations = 0
        
        # Simulate the operations step-by-step according to the rules.
        # The simulation continues as long as the current array is not non-decreasing.
        # The process stops automatically when the array becomes non-decreasing,
        # which includes the case when the array length is reduced to 1.
        while not is_non_decreasing(nums):
            # Find the adjacent pair with the minimum sum.
            # Since the array is not non-decreasing and the loop condition
            # `while not is_non_decreasing(nums)` implies len(nums) >= 2,
            # there is at least one adjacent pair to consider.
            
            min_sum = float('inf') # Initialize min_sum to positive infinity
            min_idx = -1 # Stores the index of the first element in the pair

            # Iterate through all adjacent pairs to find the one with the minimum sum.
            # The loop range covers indices i from 0 up to len(nums) - 2,
            # allowing access to nums[i] and nums[i+1].
            for i in range(len(nums) - 1):
                current_sum = nums[i] + nums[i+1]
                
                # Compare current_sum with the minimum sum found so far.
                # Using strict inequality '<' ensures that if multiple pairs have
                # the same minimum sum, the first one encountered (leftmost) is chosen,
                # satisfying the problem requirement.
                if current_sum < min_sum:
                    min_sum = current_sum
                    min_idx = i
            
            # Perform the merge operation on the chosen pair (nums[min_idx], nums[min_idx+1]).
            # The element at min_idx is updated to the sum of the pair.
            nums[min_idx] = nums[min_idx] + nums[min_idx+1]
            # The second element of the pair is removed from the list.
            # pop() operation on a list takes O(N) time in the worst case
            # (removing from the beginning) and O(1) on average (removing from the end),
            # but for removing from the middle, it's effectively O(N) because elements
            # after the removed one need to be shifted. Given N <= 50, this is acceptable.
            nums.pop(min_idx + 1)

            # Increment the counter for the number of operations performed.
            operations += 1

        # Once the loop condition `not is_non_decreasing(nums)` is false,
        # the array is non-decreasing. Return the total number of operations
        # performed to reach this state following the deterministic rule.
        return operations