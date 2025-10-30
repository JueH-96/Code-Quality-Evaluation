from typing import List
import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def get_first_digit(n: int) -> int:
            while n >= 10:
                n = n // 10
            return n
        
        first_digits = [get_first_digit(num) for num in nums]
        last_digits = [num % 10 for num in nums]
        
        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(first_digits[i], last_digits[j]) == 1:
                    count += 1
        return count