from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)

        def is_increasing(subarray):
            return all(subarray[i] < subarray[i + 1] for i in range(len(subarray) - 1))

        for i in range(n - 2 * k + 1):
            first_subarray = nums[i:i + k]
            second_subarray = nums[i + k:i + 2 * k]
            if is_increasing(first_subarray) and is_increasing(second_subarray):
                return True

        return False