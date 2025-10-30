import math
from typing import List

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
            for j in range(i + 1, n):
                if math.gcd(first_digits[i], last_digits[j]) == 1:
                    count += 1
        return count