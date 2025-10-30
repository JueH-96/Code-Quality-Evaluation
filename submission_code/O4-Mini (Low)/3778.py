from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Replace even numbers with 0 and odd numbers with 1
        transformed = [n % 2 for n in nums]
        # Sort in non-decreasing order
        transformed.sort()
        return transformed