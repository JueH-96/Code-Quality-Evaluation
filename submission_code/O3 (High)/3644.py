from typing import List
import math

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        """
        Returns the minimum positive sum of any subarray whose length
        lies between l and r (inclusive).  If no such subarray exists
        the function returns -1.
        """
        n = len(nums)
        best = math.inf         # holds the minimum positive sum found

        # Enumerate every start index
        for start in range(n):
            current_sum = 0
            # Extend the subarray up to length r
            for end in range(start, min(n, start + r)):
                current_sum += nums[end]
                length = end - start + 1
                if length >= l and current_sum > 0:
                    best = min(best, current_sum)

        return -1 if best == math.inf else best