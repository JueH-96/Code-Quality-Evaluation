from typing import List
from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def first_digit(n):
            while n >= 10:
                n //= 10
            return n
        
        def last_digit(n):
            return n % 10
        
        beautiful_pairs = 0
        n = len(nums)
        
        for i in range(n):
            first_d = first_digit(nums[i])
            for j in range(i + 1, n):
                last_d = last_digit(nums[j])
                if gcd(first_d, last_d) == 1:
                    beautiful_pairs += 1
        
        return beautiful_pairs