from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute all indices where nums[i] is equal to x.
        indices = [i for i, num in enumerate(nums) if num == x]
        res = []
        for q in queries:
            # Convert occurrence query to zero-based index.
            if q - 1 < len(indices):
                res.append(indices[q - 1])
            else:
                res.append(-1)
        return res