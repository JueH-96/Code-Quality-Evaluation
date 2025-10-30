from typing import List
from collections import Counter

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        # Count frequency of each digit in the input array
        digit_counts = Counter(digits)

        distinct_even_numbers_count = 0

        # Iterate through all possible 3-digit even numbers
        # 3-digit numbers are from 100 to 999. Even means the last digit is 0, 2, 4, 6, 8.
        # We can iterate from 100 to 998, stepping by 2.
        for num in range(100, 1000, 2):
            # Check if the current number 'num' can be formed using the available digits

            # Extract digits of the current number
            d3 = num % 10      # Last digit
            d2 = (num // 10) % 10 # Middle digit
            d1 = num // 100     # First digit

            # Count frequency of digits in the current number
            num_digit_counts = Counter([d1, d2, d3])

            # Check if digits required for 'num' are available in 'digits' array
            is_formable = True
            for digit, count in num_digit_counts.items():
                # If the number of times a digit is needed ('count') is more than
                # the number of times it's available in the input 'digits' array
                # ('digit_counts.get(digit, 0)'), then the number is not formable.
                if digit_counts.get(digit, 0) < count:
                    is_formable = False
                    break # Not enough copies of this digit

            # If formable, increment the count
            if is_formable:
                distinct_even_numbers_count += 1

        return distinct_even_numbers_count