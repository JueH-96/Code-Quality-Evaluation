import math
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        n = len(nums)
        beautiful_pairs_count = 0

        # Helper function to get the first digit of a number.
        # For example:
        # get_first_digit(123) returns 1
        # get_first_digit(5) returns 5
        # This works by repeatedly dividing the number by 10 until it's a single digit.
        def get_first_digit(num: int) -> int:
            while num >= 10:
                num //= 10
            return num

        # Helper function to get the last digit of a number.
        # For example:
        # get_last_digit(123) returns 3
        # get_last_digit(5) returns 5
        # The problem constraints state nums[i] % 10 != 0,
        # so the last digit will always be between 1 and 9 (inclusive).
        def get_last_digit(num: int) -> int:
            return num % 10

        # Iterate through all possible pairs (i, j) such that 0 <= i < j < n.
        # The outer loop iterates through 'i' from the first element up to the second-to-last.
        for i in range(n):
            # Calculate the first digit of nums[i] once for the current 'i'.
            first_digit_i = get_first_digit(nums[i])
            
            # The inner loop iterates through 'j' starting from the element immediately after 'i'
            # up to the last element of the array.
            for j in range(i + 1, n):
                # Calculate the last digit of nums[j].
                last_digit_j = get_last_digit(nums[j])

                # Check if the first digit of nums[i] and the last digit of nums[j] are coprime.
                # Two integers are coprime if their greatest common divisor (GCD) is 1.
                if math.gcd(first_digit_i, last_digit_j) == 1:
                    beautiful_pairs_count += 1
        
        return beautiful_pairs_count