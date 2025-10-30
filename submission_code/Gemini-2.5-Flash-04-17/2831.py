from typing import List
import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # --- Code starts here ---
        n = len(nums)
        count = 0

        # Helper function to get the first digit of a positive integer.
        # Assumes num is a positive integer.
        def get_first_digit(num):
            # Convert the number to a string.
            s_num = str(num)
            # The first digit is the character at index 0 of the string, converted back to an integer.
            return int(s_num[0])

        # Iterate through all possible starting indices i.
        # A pair (i, j) must have i < j, so i can range from 0 up to n-2.
        for i in range(n - 1):
            # Get the first digit of the number at index i.
            first_digit_i = get_first_digit(nums[i])

            # Iterate through all possible ending indices j.
            # j must be greater than i, so j ranges from i + 1 up to n-1.
            for j in range(i + 1, n):
                # Get the last digit of the number at index j.
                # The last digit of a positive integer is the remainder when divided by 10.
                # The problem constraints guarantee nums[i] % 10 != 0, so last digit is 1-9.
                last_digit_j = nums[j] % 10

                # Check if the first digit of nums[i] and the last digit of nums[j] are coprime.
                # Two integers x and y are coprime if their greatest common divisor (gcd) is 1.
                # Use math.gcd() function to calculate the greatest common divisor.
                if math.gcd(first_digit_i, last_digit_j) == 1:
                    # If they are coprime, this pair (i, j) is beautiful.
                    # Increment the count of beautiful pairs.
                    count += 1

        # After checking all possible pairs (i, j) with i < j, return the total count.
        return count
        # --- Code ends here ---