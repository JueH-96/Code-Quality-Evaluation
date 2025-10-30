from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        # Helper to check if the array is non-decreasing
        def is_non_decreasing(arr: List[int]) -> bool:
            for i in range(1, len(arr)):
                if arr[i] < arr[i - 1]:
                    return False
            return True

        ops = 0
        # Keep merging until the array is non-decreasing
        while not is_non_decreasing(nums):
            # Find the adjacent pair with the minimum sum (leftmost on ties)
            min_sum = nums[0] + nums[1]
            min_idx = 0
            for i in range(1, len(nums) - 1):
                s = nums[i] + nums[i + 1]
                if s < min_sum:
                    min_sum = s
                    min_idx = i
            # Merge the pair at min_idx and min_idx+1
            # Replace them with their sum
            nums = nums[:min_idx] + [min_sum] + nums[min_idx+2:]
            ops += 1

        return ops