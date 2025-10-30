from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Step 1 & 2: map even → 0, odd → 1
        mapped = [0 if num % 2 == 0 else 1 for num in nums]
        
        # Step 3: sort in non-decreasing order
        mapped.sort()
        
        return mapped