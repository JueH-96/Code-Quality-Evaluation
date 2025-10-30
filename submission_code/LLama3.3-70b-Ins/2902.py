from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(num: int) -> int:
            """Return the maximum digit in a number."""
            return max(int(digit) for digit in str(num))

        max_sum = -1
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                # Check if the maximum digits of the two numbers are equal
                if max_digit(nums[i]) == max_digit(nums[j]):
                    # Update the maximum sum if the current sum is larger
                    max_sum = max(max_sum, nums[i] + nums[j])

        return max_sum