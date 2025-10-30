from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        """
        For each number determine if it is already divisible by 3.
        If not, it takes exactly one operation (±1) to reach the nearest multiple of 3.
        Therefore the minimum operations equal the count of numbers whose remainder
        modulo 3 is non-zero.
        """
        return sum(1 for x in nums if x % 3 != 0)