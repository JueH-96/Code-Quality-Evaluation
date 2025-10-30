from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        """
        For each number in `nums`, compute the sum of its digits and
        return the minimum of these digit–sums.
        """
        def digit_sum(n: int) -> int:
            total = 0
            while n:
                total += n % 10
                n //= 10
            return total
        
        return min(digit_sum(num) for num in nums)