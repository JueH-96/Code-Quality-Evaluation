import math
from typing import List

class Solution:
    """
    This class provides a solution to count beautiful pairs in a list of numbers.
    A pair of indices (i, j) is beautiful if 0 <= i < j < len(nums) and
    the first digit of nums[i] and the last digit of nums[j] are coprime (gcd is 1).
    """
    def countBeautifulPairs(self, nums: List[int]) -> int:
        """
        Counts the total number of beautiful pairs in the given list of numbers.

        Args:
            nums: A 0-indexed list of integers. Constraints:
                  2 <= nums.length <= 100
                  1 <= nums[i] <= 9999
                  nums[i] % 10 != 0 (last digit is never 0)

        Returns:
            The total count of beautiful pairs.
        """
        n = len(nums)
        count = 0

        # Iterate through all possible pairs of indices (i, j) such that 0 <= i < j < n.
        # The outer loop iterates from i = 0 to n-1.
        # The inner loop iterates from j = i+1 to n-1.
        # This ensures that each pair (i, j) with i < j is considered exactly once.
        for i in range(n):
            # Calculate the first digit of nums[i].
            # Convert the number to a string, take the first character, and convert it back to an integer.
            # Since 1 <= nums[i] <= 9999, nums[i] is always positive, and this operation is valid.
            # The first digit will always be between 1 and 9.
            # Example: nums[i] = 345 -> str(nums[i]) = "345" -> str(nums[i])[0] = "3" -> int("3") = 3
            first_digit_i = int(str(nums[i])[0])

            for j in range(i + 1, n):
                # Calculate the last digit of nums[j].
                # The modulo 10 operator (%) gives the last digit of a positive integer.
                # Example: nums[j] = 678 -> nums[j] % 10 = 8
                # The constraint nums[j] % 10 != 0 ensures the last digit is between 1 and 9.
                last_digit_j = nums[j] % 10

                # Check if the first digit of nums[i] and the last digit of nums[j] are coprime.
                # Two positive integers are coprime if their greatest common divisor (gcd) is 1.
                # We use the math.gcd function for this calculation.
                # Both first_digit_i and last_digit_j are guaranteed to be integers between 1 and 9.
                if math.gcd(first_digit_i, last_digit_j) == 1:
                    # If the gcd is 1, the digits are coprime, and the pair (i, j) is beautiful.
                    # Increment the counter for beautiful pairs.
                    count += 1

        # After checking all pairs, return the total count of beautiful pairs found.
        return count