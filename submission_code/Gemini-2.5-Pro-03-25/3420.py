import collections
from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        """
        Finds the index of the k-th occurrence of x for multiple queries.

        Args:
            nums: The input integer array.
            queries: An array where each element k represents a query for the k-th occurrence.
            x: The target integer to find occurrences of.

        Returns:
            An array containing the answers to each query. The answer is the 0-based index
            of the k-th occurrence, or -1 if it doesn't exist.
        """
        
        # Step 1: Precompute the indices of all occurrences of x in nums.
        # We store the indices in a list. The index of an element in this list
        # corresponds to its occurrence number (0-based). For example,
        # occurrence_indices[0] is the index of the 1st occurrence,
        # occurrence_indices[1] is the index of the 2nd occurrence, and so on.
        occurrence_indices = []
        for i, num in enumerate(nums):
            if num == x:
                occurrence_indices.append(i)
                
        # Step 2: Process each query.
        answer = []
        num_occurrences = len(occurrence_indices) # Total count of x in nums
        
        for k in queries:
            # The query k is 1-based (1st occurrence, 2nd occurrence, etc.).
            # Our occurrence_indices list is 0-based.
            # So, the k-th occurrence corresponds to index k-1 in occurrence_indices.
            
            # Check if the k-th occurrence exists.
            # k must be at least 1 (as per constraints) and at most num_occurrences.
            if 1 <= k <= num_occurrences:
                # The k-th occurrence exists. Retrieve its index from our precomputed list.
                # Remember to use k-1 because the list is 0-indexed.
                result_index = occurrence_indices[k - 1]
                answer.append(result_index)
            else:
                # The k-th occurrence does not exist (either k is too large or <= 0).
                # Since constraints say k >= 1, it must be too large.
                answer.append(-1)
                
        # Step 3: Return the results for all queries.
        return answer