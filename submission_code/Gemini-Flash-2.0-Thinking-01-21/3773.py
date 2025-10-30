from typing import List

class Solution:
    def is_non_decreasing(self, arr: List[int]) -> bool:
        """Helper function to check if the array is non-decreasing."""
        # An array with 0 or 1 element is always non-decreasing.
        # The loop `range(len(arr) - 1)` correctly handles this (empty range).
        for i in range(len(arr) - 1):
            if arr[i] > arr[i+1]:
                return False
        return True

    def minimumPairRemoval(self, nums: List[int]) -> int:
        """
        Given an array nums, perform operations to make it non-decreasing.
        Operation: Select the adjacent pair with the minimum sum (leftmost tie).
                   Replace the pair with their sum.
        Return the minimum number of operations needed.
        """
        operations = 0
        # Continue operations until the array becomes non-decreasing.
        # The process stops when the array has length 1 (always non-decreasing)
        # or when it becomes non-decreasing earlier.
        while not self.is_non_decreasing(nums):
            min_sum = float('inf') # Initialize with a value larger than any possible sum
            min_index = -1         # To store the index of the first element of the pair

            # Find the adjacent pair (nums[i], nums[i+1]) with the minimum sum.
            # Iterate through all possible adjacent pairs.
            # The loop runs for indices i from 0 up to len(nums) - 2.
            for i in range(len(nums) - 1):
                current_sum = nums[i] + nums[i+1]

                # If the current sum is strictly less than the minimum sum found so far,
                # update the minimum sum and the index of the pair.
                if current_sum < min_sum:
                    min_sum = current_sum
                    min_index = i
                # If the current sum is equal to the minimum sum found so far,
                # we do nothing because the problem requires selecting the leftmost pair
                # in case of a tie. Since we iterate from left to right, `min_index`
                # will already hold the index of the leftmost pair with that minimum sum
                # when it was first encountered.

            # After finding the leftmost pair with the minimum sum at index min_index,
            # perform the replacement operation.
            # The elements at indices min_index and min_index + 1 are replaced by their sum.
            # This operation reduces the length of the list by one.
            # We create a new list to represent the state of the array after the operation.
            # The new list is formed by concatenating three parts:
            # 1. Elements from the start of the list up to (but not including) min_index: nums[:min_index]
            # 2. A list containing the sum of the pair as a single element: [nums[min_index] + nums[min_index+1]]
            # 3. Elements from the element after the pair (at index min_index + 2) to the end: nums[min_index+2:]
            nums = nums[:min_index] + [nums[min_index] + nums[min_index+1]] + nums[min_index+2:]

            # Increment the counter for the number of operations performed.
            operations += 1

        # The while loop terminates when the array `nums` becomes non-decreasing.
        # The total number of operations performed to reach this state is the result.
        return operations