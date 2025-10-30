from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)

        # Check if the array is already sorted
        if nums == sorted(nums):
            return 0

        # Create a copy of the original list extended by itself
        extended_nums = nums * 2

        # Find the sorted version of nums
        sorted_nums = sorted(nums)

        # Check for the sorted sequence in the extended list
        for i in range(n):
            if extended_nums[i:i+n] == sorted_nums:
                return (n - i) % n

        # If no such sequence is found, return -1
        return -1