from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # First, find all occurrences of x in nums
        occurrences = [i for i, num in enumerate(nums) if num == x]
        
        # Initialize an empty list to store the answers
        answers = []
        
        # For each query, find the index of the queries[i]^th occurrence of x
        for query in queries:
            # If there are fewer than queries[i] occurrences of x, append -1 to answers
            if query > len(occurrences):
                answers.append(-1)
            # Otherwise, append the index of the queries[i]^th occurrence of x
            else:
                answers.append(occurrences[query - 1])
        
        return answers