class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        
        # To efficiently handle multiple queries for the occurrences of a single element `x`,
        # we first pre-process the `nums` array to find all indices of `x`.
        # This avoids re-scanning the `nums` array for each query.

        # Step 1: Collect all indices where `x` appears using a list comprehension.
        # The resulting list `indices_of_x` stores the indices in increasing order,
        # corresponding to the 1st, 2nd, 3rd, ... occurrences.
        indices_of_x = [i for i, num in enumerate(nums) if num == x]
        
        # Get the total number of times `x` occurs in `nums`.
        num_occurrences = len(indices_of_x)
        
        # Step 2: Answer each query using the pre-computed list of indices.
        # For a given query `q`, we are looking for the q-th occurrence.
        # Since queries are 1-indexed, this corresponds to the element at index `q-1`
        # in our `indices_of_x` list.
        
        answer = []
        for q in queries:
            # Check if the q-th occurrence exists. 
            # From constraints, q >= 1, so we only need to check the upper bound.
            if q <= num_occurrences:
                # The q-th occurrence exists, its index is at indices_of_x[q-1].
                answer.append(indices_of_x[q - 1])
            else:
                # The q-th occurrence does not exist.
                answer.append(-1)
        
        return answer