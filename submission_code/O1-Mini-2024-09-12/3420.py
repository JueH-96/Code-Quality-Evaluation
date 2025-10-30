from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute the indices where x occurs
        occurrences = []
        for index, num in enumerate(nums):
            if num == x:
                occurrences.append(index)
        
        # Process each query
        answer = []
        for q in queries:
            if 1 <= q <= len(occurrences):
                answer.append(occurrences[q - 1])
            else:
                answer.append(-1)
        
        return answer