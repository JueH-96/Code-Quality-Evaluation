from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        """
        Repeatedly merge the adjacent pair with the minimum sum (leftmost in ties)
        until the array is non-decreasing. Return the number of merges performed.
        """
        def is_non_decreasing(arr: List[int]) -> bool:
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False
            return True

        ops = 0
        # Keep merging until the array is non-decreasing
        while not is_non_decreasing(nums) and len(nums) > 1:
            # Find the adjacent pair with minimum sum
            min_sum = nums[0] + nums[1]
            min_idx = 0
            for i in range(1, len(nums) - 1):
                s = nums[i] + nums[i+1]
                if s < min_sum:
                    min_sum = s
                    min_idx = i
            # Merge that pair
            merged_val = nums[min_idx] + nums[min_idx + 1]
            nums = nums[:min_idx] + [merged_val] + nums[min_idx+2:]
            ops += 1

        return ops