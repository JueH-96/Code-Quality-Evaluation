from typing import List
import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Compute the global gcd of the array
        g = 0
        for x in nums:
            g = math.gcd(g, x)
        # Count how many elements equal to the gcd
        count_g = sum(1 for x in nums if x == g)
        # If fewer than two copies of g, we can always reduce to length 1
        if count_g < 2:
            return 1
        # Otherwise, once all values have been reduced to g,
        # we have count_g copies of g to eliminate via zero‐remainder operations.
        # Each such operation removes two g's and adds one zero.
        # The minimal final length is ceil(count_g / 2).
        return (count_g + 1) // 2