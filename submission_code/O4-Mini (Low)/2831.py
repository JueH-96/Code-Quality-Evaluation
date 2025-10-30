from typing import List
import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # Helper to get first digit
        def first_digit(x: int) -> int:
            while x >= 10:
                x //= 10
            return x
        
        # Helper to get last digit
        def last_digit(x: int) -> int:
            return x % 10
        
        n = len(nums)
        # Precompute first digits and last digits
        firsts = [first_digit(x) for x in nums]
        lasts = [last_digit(x) for x in nums]
        
        count = 0
        # Check all pairs i < j
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(firsts[i], lasts[j]) == 1:
                    count += 1
        return count