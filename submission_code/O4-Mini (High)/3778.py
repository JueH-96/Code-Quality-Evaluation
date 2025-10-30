from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Replace evens with 0 and odds with 1
        transformed = [0 if x % 2 == 0 else 1 for x in nums]
        # Sort in non-decreasing order
        transformed.sort()
        return transformed