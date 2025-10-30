from typing import List

class Solution:
    # Helper method to calculate the sum of digits for a number.
    # Marked as static as it doesn't use 'self' (instance-specific data).
    @staticmethod
    def _calculate_sum_of_digits(n: int) -> int:
        s = 0
        # According to constraints, n is always a positive integer (1 <= n <= 10^4).
        # This loop correctly calculates the sum of digits for positive integers.
        # Example: n=123
        # 1. s = 0 + 123 % 10 = 3, n = 123 // 10 = 12
        # 2. s = 3 + 12 % 10 = 3 + 2 = 5, n = 12 // 10 = 1
        # 3. s = 5 + 1 % 10 = 5 + 1 = 6, n = 1 // 10 = 0
        # Loop ends, returns 6.
        while n > 0:
            s += n % 10  # Add the last digit to sum
            n //= 10     # Remove the last digit from n
        return s

    def minElement(self, nums: List[int]) -> int:
        # Problem constraints ensure:
        # - nums is not empty (1 <= nums.length <= 100).
        # - Elements of nums are positive integers (1 <= nums[i] <= 10^4).

        # Create a generator that yields the sum of digits for each number in nums.
        # The _calculate_sum_of_digits method is static. In Python, static methods
        # can be called via the class name (Solution._calculate_sum_of_digits) or
        # via an instance (self._calculate_sum_of_digits). Both are valid.
        
        # Example: nums = [10, 12, 13, 14]
        # _calculate_sum_of_digits(10) -> 1
        # _calculate_sum_of_digits(12) -> 3
        # _calculate_sum_of_digits(13) -> 4
        # _calculate_sum_of_digits(14) -> 5
        # The generator will yield 1, 3, 4, 5.
        digit_sums_generator = (self._calculate_sum_of_digits(num) for num in nums)
        
        # The min() function finds the minimum value from the sequence provided by the generator.
        # For the example above, min(1, 3, 4, 5) -> 1.
        # Since nums is guaranteed to be non-empty, the generator will also be non-empty,
        # so min() will always receive at least one value.
        return min(digit_sums_generator)