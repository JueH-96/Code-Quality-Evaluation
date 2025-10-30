from typing import List
from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        beautiful_count = 0
        
        # Function to get the first digit of a number
        def first_digit(n):
            while n >= 10:
                n //= 10
            return n
        
        for i in range(len(nums)):
            first = first_digit(nums[i])
            for j in range(i + 1, len(nums)):
                last = nums[j] % 10
                if gcd(first, last) == 1:
                    beautiful_count += 1
        
        return beautiful_count