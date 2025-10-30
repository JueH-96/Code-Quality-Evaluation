from typing import List
import math

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # 1. Build the first_occurrence dictionary
        # Map value to its smallest index in elements.
        # We need the smallest index because if multiple elements can be assigned,
        # the one with the smallest index j is chosen.
        first_occurrence = {}
        for j, value in enumerate(elements):
            # We only care about the first occurrence of each value.
            if value not in first_occurrence:
                first_occurrence[value] = j

        # 2. Initialize assigned array
        n_groups = len(groups)
        n_elements = len(elements)
        # Initialize all assignments to -1, indicating no element found yet for any group.
        assigned = [-1] * n_groups

        # 3. Iterate through each group to find a suitable element
        for i in range(n_groups):
            group_value = groups[i]
            # Initialize min_j with a value larger than any valid index (0 to n_elements-1).
            # This variable will store the smallest index j of an element that can be
            # assigned to the current group. If it remains n_elements, no suitable
            # element is found.
            min_j = n_elements

            # Find divisors of the current group_value and check their first occurrence index
            # in the elements array.
            # We only need to iterate through potential divisors up to the square root
            # of the group_value. If k is a divisor of G, then G // k is also a divisor.
            # If k <= sqrt(G), then G // k >= sqrt(G). By checking both k and G // k,
            # we find all pairs of divisors.
            # We use math.sqrt to calculate the square root.
            # math.floor gets the largest integer less than or equal to the sqrt.
            # We add 1 to the floor result for the range upper bound to include the
            # integer square root itself if the group_value is a perfect square.
            sqrt_g = int(math.floor(math.sqrt(group_value)))

            # Iterate through potential divisors k from 1 up to sqrt_g.
            for k in range(1, sqrt_g + 1):
                # Check if k divides the group_value.
                if group_value % k == 0:
                    d1 = k             # The first divisor found
                    d2 = group_value // k # The paired divisor

                    # Check if the first divisor (d1) exists in the elements array
                    # using the pre-calculated first_occurrence map.
                    if d1 in first_occurrence:
                        # Get the smallest index of d1 in elements.
                        j1 = first_occurrence[d1]
                        # Update min_j if this index is smaller than the current minimum.
                        min_j = min(min_j, j1)

                    # Check if the second divisor (d2) exists in elements.
                    # We only check d2 if it's different from d1. This avoids
                    # redundant checks when group_value is a perfect square (d1 == d2).
                    # If d1 != d2, d2 must be greater than or equal to sqrt(group_value).
                    if d1 != d2 and d2 in first_occurrence:
                         # Get the smallest index of d2 in elements.
                         j2 = first_occurrence[d2]
                         # Update min_j if this index is smaller than the current minimum.
                         min_j = min(min_j, j2)

            # After checking all relevant divisors (derived from pairs (k, G/k) where k <= sqrt(G)),
            # min_j holds the smallest index of an element that divides group_value,
            # or it remains n_elements if no such element was found in the elements array.
            
            # If a valid index was found (i.e., min_j is less than n_elements), assign it.
            if min_j < n_elements:
                assigned[i] = min_j
            # If min_j remains n_elements, it means no suitable element was found,
            # and assigned[i] correctly remains its initial value of -1.

        # Return the resulting array of assigned element indices.
        return assigned