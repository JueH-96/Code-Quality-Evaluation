from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        
        x_indices = []
        # Step 1: Collect all indices where x appears in nums.
        # These indices are stored in x_indices, naturally in ascending order
        # because we iterate through `nums` from left to right.
        for index, value in enumerate(nums):
            if value == x:
                x_indices.append(index)
        
        # `num_occurrences` stores the total count of x in `nums`.
        num_occurrences = len(x_indices)
        
        # Step 2: Prepare the list to store answers for each query.
        # The problem asks to "Return an integer array answer".
        answer = []
        
        # Process each query in `queries`.
        for q_val in queries:
            # `q_val` is the 1-indexed occurrence number we are looking for.
            # For example, if q_val is 1, we are looking for the 1st occurrence of x.
            # The problem constraints state 1 <= queries[i] <= 10^5, so q_val is always >= 1.
            
            # Check if the requested q_val-th occurrence actually exists.
            # This is true if q_val is less than or equal to the total number of occurrences.
            if q_val <= num_occurrences:
                # If it exists, its actual index in the `nums` array is stored in `x_indices`.
                # Since `x_indices` is 0-indexed, the q_val-th item is at `x_indices[q_val - 1]`.
                answer.append(x_indices[q_val - 1])
            else:
                # If `q_val` is greater than `num_occurrences`,
                # then the q_val-th occurrence does not exist in `nums`.
                answer.append(-1)
                
        return answer