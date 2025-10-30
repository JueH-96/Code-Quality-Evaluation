from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Collect all indices where nums[i] == x
        x_indices = [i for i, num in enumerate(nums) if num == x]
        
        # Prepare the result list
        result = []
        for q in queries:
            if q > len(x_indices):
                result.append(-1)
            else:
                result.append(x_indices[q - 1])
        
        return result