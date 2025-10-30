from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all the indices where x occurs in nums
        occurrences = []
        for index, num in enumerate(nums):
            if num == x:
                occurrences.append(index)
        
        # Prepare the answer for each query
        answer = []
        for query in queries:
            if query <= len(occurrences):
                answer.append(occurrences[query - 1])
            else:
                answer.append(-1)
        
        return answer