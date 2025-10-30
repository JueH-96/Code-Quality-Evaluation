from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute all indices where nums[i] == x
        positions = []
        for i, val in enumerate(nums):
            if val == x:
                positions.append(i)
        
        # Answer each query by checking if the k-th occurrence exists
        result = []
        for k in queries:
            # k-th occurrence is at positions[k-1] if it exists
            if k <= len(positions):
                result.append(positions[k-1])
            else:
                result.append(-1)
        
        return result