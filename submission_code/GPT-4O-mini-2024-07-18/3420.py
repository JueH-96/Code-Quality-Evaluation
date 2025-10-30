from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all occurrences of x in nums
        occurrences = [i for i, num in enumerate(nums) if num == x]
        
        # Prepare the result for each query
        result = []
        for query in queries:
            if query <= len(occurrences):
                result.append(occurrences[query - 1])  # query is 1-indexed
            else:
                result.append(-1)
        
        return result