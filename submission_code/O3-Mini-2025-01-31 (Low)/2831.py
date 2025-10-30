from math import gcd
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def first_digit(x: int) -> int:
            # Keep dividing x by 10 until it's less than 10.
            while x >= 10:
                x //= 10
            return x
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            first = first_digit(nums[i])
            for j in range(i+1, n):
                last = nums[j] % 10
                # Check if the first digit of nums[i] and the last digit of nums[j] are coprime.
                if gcd(first, last) == 1:
                    count += 1
                    
        return count