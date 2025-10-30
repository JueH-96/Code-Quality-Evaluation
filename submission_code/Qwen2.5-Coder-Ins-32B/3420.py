from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a list to store the indices of occurrences of x in nums
        occurrences = []
        
        # Iterate through nums and collect indices where nums[i] == x
        for i, num in enumerate(nums):
            if num == x:
                occurrences.append(i)
        
        # Prepare the answer list
        answer = []
        
        # For each query, find the index of the query^th occurrence of x
        for query in queries:
            if query <= len(occurrences):
                answer.append(occurrences[query - 1])
            else:
                answer.append(-1)
        
        return answer