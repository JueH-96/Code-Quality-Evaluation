from typing import List
from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # Pre-compute first and last digits for every number
        first_digits = [int(str(x)[0]) for x in nums]   # first digit of each number
        last_digits  = [x % 10 for x in nums]           # last digit of each number
        
        n = len(nums)
        beautiful = 0
        
        # Count pairs (i, j) with 0 ≤ i < j < n
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(first_digits[i], last_digits[j]) == 1:
                    beautiful += 1
                    
        return beautiful