from typing import List
from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        frequency = [0] * 10  # index 0 is unused
        total = 0
        for num in nums:
            # Compute first digit
            n = num
            while n >= 10:
                n = n // 10
            first = n
            last = num % 10
            # Calculate sum for current last digit
            current_sum = 0
            for d in range(1, 10):
                if gcd(d, last) == 1:
                    current_sum += frequency[d]
            total += current_sum
            # Update frequency with current first digit
            frequency[first] += 1
        return total