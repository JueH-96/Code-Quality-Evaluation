from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        found = False

        # Iterate over all possible subarray lengths from l to r
        for length in range(l, r + 1):
            current_sum = sum(nums[:length])
            if current_sum > 0:
                min_sum = min(min_sum, current_sum)
                found = True

            # Sliding window to find the minimum sum of the current length
            for i in range(length, len(nums)):
                current_sum += nums[i] - nums[i - length]
                if current_sum > 0:
                    min_sum = min(min_sum, current_sum)
                    found = True

        return min_sum if found else -1