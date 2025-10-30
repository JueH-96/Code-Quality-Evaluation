from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Collect all indices where nums[i] == x
        indices = []
        for idx, num in enumerate(nums):
            if num == x:
                indices.append(idx)
        
        # Process each query
        result = []
        for q in queries:
            if q > len(indices):
                result.append(-1)
            else:
                result.append(indices[q - 1])
        
        return result