from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        """
        For each number, compute its remainder mod 3. The cost to make it divisible
        by 3 is the smaller of (remainder) and (3 - remainder), since you can either
        decrement or increment to the nearest multiple of 3.
        """
        operations = 0
        for num in nums:
            r = num % 3
            operations += min(r, 3 - r)
        return operations