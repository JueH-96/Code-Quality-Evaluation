from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        """
        Finds the index of the k-th occurrence of x in nums for each query.

        Args:
            nums: A list of integers.
            queries: A list of integers, where queries[i] represents the k for
                     the k-th occurrence.
            x: The integer whose occurrences are to be found.

        Returns:
            An integer array answer, where answer[i] is the index of the
            queries[i]-th occurrence of x, or -1 if there are fewer than
            queries[i] occurrences of x.
        """

        # Step 1: Precompute all indices where 'x' occurs in the 'nums' array.
        # This list will store the 0-based indices of all occurrences of 'x'.
        x_indices = []
        for i, num in enumerate(nums):
            if num == x:
                x_indices.append(i)
        
        # Get the total number of occurrences of 'x'. This will be used to
        # check if a queried occurrence exists.
        total_occurrences = len(x_indices)
        
        # Step 2: Process each query.
        # Initialize an empty list to store the results for each query.
        answer = []
        
        for query_val in queries:
            # 'query_val' represents the k-th occurrence (1-indexed).
            # To access the corresponding index in 'x_indices' (0-indexed),
            # we need to use 'query_val - 1'.

            # Check if the requested occurrence (query_val) is within the bounds
            # of the actual occurrences of 'x'.
            if query_val > total_occurrences:
                # If 'query_val' is greater than the total number of occurrences,
                # it means the requested occurrence does not exist.
                answer.append(-1)
            else:
                # Otherwise, the occurrence exists. Retrieve its index from
                # the precomputed 'x_indices' list.
                answer.append(x_indices[query_val - 1])
                
        # Step 3: Return the accumulated answers.
        return answer