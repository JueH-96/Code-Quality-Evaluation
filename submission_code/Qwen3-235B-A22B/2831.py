import math
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        first_digits = [int(str(num)[0]) for num in nums]
        last_digits = [num % 10 for num in nums]
        
        for i in range(n):
            for j in range(i + 1, n):
                a = first_digits[i]
                b = last_digits[j]
                if math.gcd(a, b) == 1:
                    count += 1
        return count