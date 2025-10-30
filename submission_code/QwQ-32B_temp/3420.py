from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        x_indices = []
        # Collect all indices where nums[i] == x
        for idx, num in enumerate(nums):
            if num == x:
                x_indices.append(idx)
        
        # Process each query
        result = []
        for q in queries:
            if q <= len(x_indices):
                result.append(x_indices[q - 1])
            else:
                result.append(-1)
        return result