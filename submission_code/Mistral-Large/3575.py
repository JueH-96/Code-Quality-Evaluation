from itertools import combinations
from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        max_value = 0
        n = len(nums)

        # Generate all possible subsequences of size 2 * k
        for subsequence in combinations(nums, 2 * k):
            # Split the subsequence into two halves
            first_half = subsequence[:k]
            second_half = subsequence[k:]

            # Calculate the OR of the first half
            or_first_half = 0
            for num in first_half:
                or_first_half |= num

            # Calculate the OR of the second half
            or_second_half = 0
            for num in second_half:
                or_second_half |= num

            # Calculate the value of the subsequence
            value = or_first_half ^ or_second_half

            # Update the maximum value
            max_value = max(max_value, value)

        return max_value