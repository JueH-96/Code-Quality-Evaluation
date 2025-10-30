from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute all indices where nums[i] == x
        positions = []
        for i, val in enumerate(nums):
            if val == x:
                positions.append(i)
        
        # For each query, retrieve the (q-th) occurrence index or -1 if not enough occurrences
        answer = []
        for q in queries:
            if 1 <= q <= len(positions):
                answer.append(positions[q - 1])
            else:
                answer.append(-1)
        
        return answer