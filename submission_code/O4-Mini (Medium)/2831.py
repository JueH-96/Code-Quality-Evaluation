from typing import List
import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # Helper to get the first digit of a positive integer
        def first_digit(x: int) -> int:
            while x >= 10:
                x //= 10
            return x

        n = len(nums)
        # Precompute first digits and last digits
        firsts = [first_digit(x) for x in nums]
        lasts  = [x % 10 for x in nums]

        count = 0
        # Count pairs (i, j) with i < j where gcd(firsts[i], lasts[j]) == 1
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(firsts[i], lasts[j]) == 1:
                    count += 1
        return count