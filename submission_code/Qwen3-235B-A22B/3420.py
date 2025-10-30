from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Collect all indices where nums[i] == x
        positions = [i for i, num in enumerate(nums) if num == x]
        
        # Process each query to find the k-th occurrence
        result = []
        for q in queries:
            if q > len(positions):
                result.append(-1)
            else:
                result.append(positions[q - 1])
        return result