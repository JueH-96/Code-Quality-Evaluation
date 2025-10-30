import math
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        first_digits = []
        last_digits = []
        for num in nums:
            # Calculate last digit
            last = num % 10
            # Calculate first digit
            first = num
            while first >= 10:
                first = first // 10
            first_digits.append(first)
            last_digits.append(last)
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(first_digits[i], last_digits[j]) == 1:
                    count += 1
        return count