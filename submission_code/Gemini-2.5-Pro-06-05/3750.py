import collections
from typing import List

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        """
        Calculates the minimum circular distance to an element with the same value
        for a series of queries.

        The solution uses a pre-computation strategy for efficiency.
        Time complexity: O(N + Q), where N is len(nums) and Q is len(queries).
        Space complexity: O(N).
        """
        n = len(nums)
        
        # Step 1: Group indices by the value they hold.
        # The lists of indices will be sorted by construction.
        value_to_indices = collections.defaultdict(list)
        for i, num in enumerate(nums):
            value_to_indices[num].append(i)
            
        # Step 2: Pre-compute the minimum distance for every index.
        # Initialize with -1, the default answer for unique elements.
        min_dists = [-1] * n
        
        # Iterate through each group of indices for a given value.
        for indices in value_to_indices.values():
            k = len(indices)
            
            # If a value appears more than once, calculate distances.
            if k > 1:
                # For each index in the group, find its closest neighbor.
                for i in range(k):
                    curr_idx = indices[i]
                    
                    # The list of indices is treated as circular.
                    # In Python, `indices[i-1]` gracefully handles the wrap-around for `i=0`.
                    prev_idx = indices[i-1] 
                    next_idx = indices[(i + 1) % k]
                    
                    # Calculate circular distance to the 'previous' neighbor.
                    dist_to_prev = abs(curr_idx - prev_idx)
                    dist_to_prev = min(dist_to_prev, n - dist_to_prev)
                    
                    # Calculate circular distance to the 'next' neighbor.
                    dist_to_next = abs(curr_idx - next_idx)
                    dist_to_next = min(dist_to_next, n - dist_to_next)
                    
                    # Store the minimum of the two distances.
                    min_dists[curr_idx] = min(dist_to_prev, dist_to_next)
        
        # Step 3: Use the pre-computed distances to answer the queries.
        return [min_dists[q] for q in queries]