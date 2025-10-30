from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a list to store the indices of x in nums
        indices = [i for i, num in enumerate(nums) if num == x]
        
        # Create a list to store the answers to the queries
        answer = []
        
        # Process each query
        for query in queries:
            if query <= len(indices):
                # If the query is within the number of occurrences, append the index
                answer.append(indices[query - 1])
            else:
                # If the query is greater than the number of occurrences, append -1
                answer.append(-1)
        
        return answer