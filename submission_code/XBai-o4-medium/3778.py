from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        modified = [0 if num % 2 == 0 else 1 for num in nums]
        return sorted(modified)