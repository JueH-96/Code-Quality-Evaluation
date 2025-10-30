from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Sort the unique values in nums in descending order
        unique_nums = sorted(set(nums), reverse=True)

        # If k is not in nums, return -1
        if k not in nums:
            return -1

        # Initialize the count of operations
        operations = 0

        # Iterate through the unique values
        for h in unique_nums:
            if h > k:
                # Perform the operation for all elements greater than h
                for i in range(len(nums)):
                    if nums[i] > h:
                        nums[i] = h
                operations += 1
            elif h == k:
                # If h is equal to k, we can stop here
                break

        # Check if all elements are equal to k
        if all(num == k for num in nums):
            return operations
        else:
            return -1