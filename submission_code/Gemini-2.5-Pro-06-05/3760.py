from collections import defaultdict
from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        """
        Assigns elements to groups based on divisibility, prioritizing smaller element indices.
        """
        if not groups:
            return []

        n = len(groups)
        assigned = [-1] * n

        # Pre-process groups: map group sizes to their indices for quick lookup.
        # At the same time, find the maximum group size to bound the search later.
        group_indices = defaultdict(list)
        max_group_size = 0
        for i, g in enumerate(groups):
            group_indices[g].append(i)
            if g > max_group_size:
                max_group_size = g

        # Use a set to process each unique element value only once.
        seen_elements = set()
        
        # Iterate through elements with their original index `j`.
        for j, v in enumerate(elements):
            if v in seen_elements:
                continue
            seen_elements.add(v)

            # Optimization: an element's value cannot be greater than a group's size
            # if it is to be a divisor of that size.
            if v > max_group_size:
                continue

            # Sieve-like approach: for the current element `v`, iterate through all its 
            # multiples `m` up to the maximum group size.
            for m in range(v, max_group_size + 1, v):
                # Check if any groups have the size `m`.
                if m in group_indices:
                    # Assign the current element `j` to all unassigned groups of size `m`.
                    for i in group_indices[m]:
                        # Because we iterate through `j` in increasing order, the first
                        # assignment we find for any group is the one with the smallest index.
                        if assigned[i] == -1:
                            assigned[i] = j
                            
        return assigned