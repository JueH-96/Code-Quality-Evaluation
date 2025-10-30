from typing import List
from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        first_digits = []
        last_digits = []
        for num in nums:
            s = str(num)
            first = int(s[0])
            last = num % 10
            first_digits.append(first)
            last_digits.append(last)
        
        count = 0
        n = len(nums)
        for i in range(n):
            a = first_digits[i]
            for j in range(i + 1, n):
                b = last_digits[j]
                if gcd(a, b) == 1:
                    count += 1
        return count