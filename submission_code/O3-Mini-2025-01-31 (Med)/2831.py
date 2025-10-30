from math import gcd
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # Function to get the first digit of a number
        def get_first_digit(num: int) -> int:
            while num >= 10:
                num //= 10
            return num
        
        n = len(nums)
        beautiful_count = 0
        
        # Precompute first digits and last digits for all numbers
        first_digits = [get_first_digit(num) for num in nums]
        last_digits = [num % 10 for num in nums]
        
        # Iterate over all pairs (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Check if first digit of nums[i] and last digit of nums[j] are coprime
                if gcd(first_digits[i], last_digits[j]) == 1:
                    beautiful_count += 1
        
        return beautiful_count