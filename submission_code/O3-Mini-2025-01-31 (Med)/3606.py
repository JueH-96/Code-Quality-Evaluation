from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        def digit_sum(n: int) -> int:
            s = 0
            while n > 0:
                s += n % 10
                n //= 10
            return s
        
        # Replace each element with its digit sum.
        replaced = [digit_sum(num) for num in nums]
        # Return the minimum element from the replaced list.
        return min(replaced)