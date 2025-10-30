from typing import List


class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # First, sort the list to easily identify the median
        nums.sort()

        # Calculate the target index (position) of the median
        mid = len(nums) // 2

        # Calculate total operations to bring all numbers up to k
        total_up = sum(k - n if n < k else 0 for n in nums)

        # Calculate total operations to bring all numbers down to k
        total_down = sum(n - k for n in nums if k < n)

        # If the median is already at k, no operation is needed
        if nums[mid] == k:
            return 0

        # If the median is to the left of k, calculate from the median to k
        if nums[mid] < k:
            return total_up - sum(n - k for n in nums[mid : mid + 1 + len(nums) % 2])

        # If the median is to the right of k, calculate from the median to k
        return total_down - sum(k - n for n in nums[mid : mid - 1 + len(nums) % 2 : -1])