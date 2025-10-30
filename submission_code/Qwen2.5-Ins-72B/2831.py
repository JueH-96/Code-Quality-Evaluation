from math import gcd
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def first_digit(n):
            while n >= 10:
                n //= 10
            return n
        
        def last_digit(n):
            return n % 10
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(first_digit(nums[i]), last_digit(nums[j])) == 1:
                    count += 1
        return count