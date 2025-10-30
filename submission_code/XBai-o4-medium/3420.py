from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        positions = []
        for idx, num in enumerate(nums):
            if num == x:
                positions.append(idx)
        
        result = []
        for q in queries:
            if q > len(positions):
                result.append(-1)
            else:
                result.append(positions[q - 1])
        return result