from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a list to store the indices of occurrences of x
        occurrences = [i for i, num in enumerate(nums) if num == x]

        # Initialize the result list
        result = []

        # Process each query
        for query in queries:
            if query <= len(occurrences):
                result.append(occurrences[query - 1])
            else:
                result.append(-1)

        return result