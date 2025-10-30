import collections
import bisect
from typing import List

# The List type hint requires importing List from typing
# Make sure it's available in the execution environment, 
# or add 'from typing import List' if needed outside a platform
# that handles it automatically.

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        """
        Calculates the minimum circular distance for each query in a circular array.

        For each query index `queries[i]`, it finds the minimum circular distance 
        to another index `j` such that `nums[j] == nums[queries[i]]`. If no such
        other index exists, the result for that query is -1.

        Args:
            nums: The circular array of numbers. Its length is n.
            queries: A list of indices to query.

        Returns:
            A list `answer` of the same size as `queries`, where `answer[i]` 
            is the minimum circular distance for `queries[i]`, or -1 if no 
            other element has the same value.
        """
        n = len(nums)
        
        # Step 1: Precompute a mapping from each value in nums to a list of indices
        # where that value appears. Using collections.defaultdict(list) makes it easy
        # to append indices without checking if the key exists.
        value_to_indices = collections.defaultdict(list)
        for i, num in enumerate(nums):
            # The indices are added in increasing order, so each list will be sorted.
            value_to_indices[num].append(i)

        # Initialize the list to store the results for each query.
        answer = []
        
        # Step 2: Process each query.
        for q_idx in queries:
            # Get the value at the current query index.
            val = nums[q_idx]
            # Retrieve the precomputed list of indices where this value appears.
            indices = value_to_indices[val]
            # Get the total number of occurrences of this value.
            k = len(indices)

            # Step 3: Handle the case where the value appears only once.
            # If k=1, the element at q_idx is unique (or the only one with its value).
            # Thus, there's no other index j with the same value.
            if k == 1:
                answer.append(-1)
                # Skip the rest of the loop and proceed to the next query.
                continue 

            # Step 4: If the value appears multiple times (k > 1).
            # Find the position 'p' of the current query index 'q_idx' within the 
            # sorted list 'indices'. We use binary search (bisect_left) for efficiency.
            # Since q_idx is guaranteed to be in 'indices' (as k > 1), 
            # indices[p] will be equal to q_idx.
            p = bisect.bisect_left(indices, q_idx)
            
            # The minimum distance must be to one of the immediate neighbors in the 
            # sorted list of indices, considering the circular nature.
            # These neighbors are at indices (p-1+k)%k and (p+1)%k in the 'indices' list.

            # Calculate the distance to the 'previous' neighbor index.
            # The modulo arithmetic correctly handles wrap-around: if p=0, the previous index is k-1.
            prev_p_in_indices = (p - 1 + k) % k
            prev_idx = indices[prev_p_in_indices]
            # Calculate the absolute linear difference between q_idx and its previous neighbor.
            diff_prev = abs(q_idx - prev_idx)
            # Calculate the circular distance in the 'nums' array of size n. 
            # This is the minimum of the direct path distance and the wrap-around path distance.
            dist_prev = min(diff_prev, n - diff_prev)

            # Calculate the distance to the 'next' neighbor index.
            # The modulo arithmetic correctly handles wrap-around: if p=k-1, the next index is 0.
            next_p_in_indices = (p + 1) % k
            next_idx = indices[next_p_in_indices]
            # Calculate the absolute linear difference between q_idx and its next neighbor.
            diff_next = abs(q_idx - next_idx)
            # Calculate the circular distance in the 'nums' array.
            dist_next = min(diff_next, n - diff_next)
            
            # The minimum circular distance required for the query q_idx is the smaller 
            # of the distances to these two neighbors.
            min_dist = min(dist_prev, dist_next)
            
            # Append the calculated minimum distance to the result list.
            answer.append(min_dist)

        # Step 5: Return the list containing the results for all queries.
        return answer