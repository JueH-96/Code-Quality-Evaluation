from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Convert each element to 0 if even, else 1
        modified = [0 if num % 2 == 0 else 1 for num in nums]
        # Sort the modified array in non-decreasing order
        modified.sort()
        return modified