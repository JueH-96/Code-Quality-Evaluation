from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        # Helper to compute the sum of digits of a number
        def digit_sum(n: int) -> int:
            s = 0
            while n:
                s += n % 10
                n //= 10
            return s
        
        # Compute digit sums for all numbers and return the minimum
        return min(digit_sum(n) for n in nums)