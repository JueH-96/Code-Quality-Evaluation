from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        """
        Replace each element in nums with the sum of its digits
        and return the minimum of those new values.
        """
        def digit_sum(n: int) -> int:
            s = 0
            while n > 0:
                s += n % 10
                n //= 10
            return s
        
        # Compute the digit sum for each number
        transformed = [digit_sum(x) for x in nums]
        # Return the minimum
        return min(transformed)