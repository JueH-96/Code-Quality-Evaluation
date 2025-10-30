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
        # Keep applying the mandated operation until sorted
        while not is_non_decreasing(nums):
            n = len(nums)
            # Find the leftmost adjacent pair with minimum sum
            min_sum = nums[0] + nums[1]
            idx = 0
            for i in range(1, n - 1):
                s = nums[i] + nums[i + 1]
                if s < min_sum:
                    min_sum = s
                    idx = i
            # Replace that pair with their sum
            nums[idx:idx+2] = [min_sum]
            ops += 1

        return ops