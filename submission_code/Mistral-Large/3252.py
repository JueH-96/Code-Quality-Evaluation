from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        incremovable_count = 0

        # Helper function to check if an array is strictly increasing
        def is_strictly_increasing(arr: List[int]) -> bool:
            return all(arr[i] < arr[i + 1] for i in range(len(arr) - 1))

        # Generate all possible subarrays
        for start in range(n):
            for end in range(start, n):
                subarray = nums[start:end + 1]
                remaining = nums[:start] + nums[end + 1:]
                if is_strictly_increasing(remaining):
                    incremovable_count += 1

        return incremovable_count