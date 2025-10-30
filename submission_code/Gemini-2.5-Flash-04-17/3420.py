from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Pre-process nums to find all indices where x occurs
        # Store these indices in a list in ascending order
        x_indices = []
        for i in range(len(nums)):
            if nums[i] == x:
                x_indices.append(i)

        # Process each query
        answer = []
        for k in queries:
            # The k-th occurrence corresponds to the element at index k-1 in the x_indices list.
            # queries[i] are 1-based (1st, 2nd, etc.), while list indices are 0-based.
            target_index_in_x_indices = k - 1

            # Check if the target index is valid (i.e., if the k-th occurrence exists)
            # The index must be non-negative (guaranteed by k >= 1 from constraints)
            # and less than the total number of occurrences (len(x_indices)).
            if target_index_in_x_indices < len(x_indices):
                # If valid, append the index of the k-th occurrence from the pre-computed list
                answer.append(x_indices[target_index_in_x_indices])
            else:
                # If the target index is out of bounds, the k-th occurrence does not exist
                answer.append(-1)

        return answer