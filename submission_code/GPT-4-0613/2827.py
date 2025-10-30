from typing import List
from math import gcd

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        # If there is a number that is not a prime number, we can traverse all pairs.
        for num in nums:
            if num != 2 and num % 2 == 0:
                return True
            i = 3
            while i * i <= num:
                if num % i:
                    i += 2
                else:
                    return True
        return False