from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single_digit = 0
        sum_double_digit = 0

        # Iterate through the numbers to categorize and sum them
        for num in nums:
            if num < 10:  # Numbers 1-9 are single-digit (given 1 <= nums[i] <= 99)
                sum_single_digit += num
            else:  # Numbers 10-99 are double-digit
                sum_double_digit += num
        
        # Alice can win if the sum of single-digit numbers is strictly greater
        # than the sum of double-digit numbers (meaning she picks single-digit numbers),
        # OR if the sum of double-digit numbers is strictly greater
        # than the sum of single-digit numbers (meaning she picks double-digit numbers).
        # This simplifies to checking if the two sums are not equal.
        return sum_single_digit != sum_double_digit