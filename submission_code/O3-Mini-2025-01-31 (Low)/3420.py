from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute all indices where element x occurs in nums.
        positions = []
        for index, value in enumerate(nums):
            if value == x:
                positions.append(index)
        
        # Prepare the result list
        results = []
        for query in queries:
            # queries are 1-indexed for occurrence count.
            if query <= len(positions):
                results.append(positions[query - 1])
            else:
                results.append(-1)
                
        return results