import math
from itertools import combinations
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        """
        Counts the number of beautiful pairs in the given list of numbers.

        A pair of indices (i, j) is beautiful if 0 <= i < j < len(nums) and
        the first digit of nums[i] and the last digit of nums[j] are coprime.
        """
        
        count = 0
        n = len(nums)
        
        # Use itertools.combinations to generate all unique pairs of indices (i, j)
        # where i < j. This is a clean and Pythonic way to iterate through pairs.
        for i, j in combinations(range(n), 2):
            # Get the first digit of nums[i].
            # Converting to a string is a simple way to extract the first digit.
            first_digit = int(str(nums[i])[0])
            
            # Get the last digit of nums[j] using the modulo operator.
            last_digit = nums[j] % 10
            
            # Two numbers are coprime if their greatest common divisor (GCD) is 1.
            if math.gcd(first_digit, last_digit) == 1:
                count += 1
                
        return count