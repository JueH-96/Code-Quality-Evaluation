from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Collect all indices where nums[i] == x
        positions = []
        for idx, val in enumerate(nums):
            if val == x:
                positions.append(idx)
        
        # For each query, retrieve the (k-1)th index if it exists
        answers = []
        total_occurrences = len(positions)
        for k in queries:
            if 1 <= k <= total_occurrences:
                answers.append(positions[k - 1])
            else:
                answers.append(-1)
        
        return answers