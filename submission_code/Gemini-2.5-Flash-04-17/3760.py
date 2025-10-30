from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        """
        Assigns elements to groups based on divisibility and smallest index rules.

        Args:
            groups: An integer array where groups[i] is the size of the i^th group.
            elements: An integer array of available elements.

        Returns:
            An integer array assigned, where assigned[i] is the index of the
            element chosen for group i, or -1 if no suitable element exists.
        """
        # Constraints: 1 <= groups[i] <= 10^5, 1 <= elements[i] <= 10^5
        # The maximum possible group value is 10^5. We can precompute
        # for each possible group value v (from 1 to 10^5), the smallest
        # index j such that elements[j] divides v.

        max_group_val = 100000 # Maximum possible value for a group size

        # min_index_divisor[v] will store the smallest index j such that elements[j] divides v.
        # Initialize with -1, indicating no suitable element has been found yet for value v.
        min_index_divisor = [-1] * (max_group_val + 1)

        # Iterate through the elements array by index j.
        # Since we need the *smallest* index j, processing elements in increasing order of j
        # ensures that the first element found for a given multiple is the one with the smallest index.
        for j in range(len(elements)):
            element_val = elements[j]

            # Optimization: If the element value is greater than the maximum
            # possible group value, it cannot divide any group value in the
            # range [1, max_group_val], so we can skip it.
            # Constraint 1 <= elements[i] ensures element_val >= 1.
            if element_val > max_group_val:
                continue

            # Iterate through multiples of the current element_val.
            # For each multiple 'multiple', elements[j] is a divisor.
            # We want to find the smallest j for each 'multiple'.
            # Since we iterate j from 0 upwards, the first time we
            # encounter a divisor for 'multiple', it will have the smallest index j so far.
            # Any subsequent divisor found (with index j' > j) for the same multiple
            # will not overwrite the entry because we only update if the new index j is smaller
            # (which won't be the case, or if the entry was -1).
            # So, the check `if min_index_divisor[multiple] == -1` is sufficient here
            # because we process elements in order of increasing index j.
            
            # Start from the first multiple (1 * element_val)
            multiple = element_val
            while multiple <= max_group_val:
                # If we haven't found a divisor for this multiple yet,
                # or if the current element index j is smaller than the one currently stored.
                # Because we iterate through elements by increasing index j, the condition
                # `j < min_index_divisor[multiple]` check is implicitly handled if we only update
                # when min_index_divisor[multiple] is -1. However, explicitly checking for -1
                # and then potentially updating if a smaller index is found is robust.
                # The logic `if min_index_divisor[multiple] == -1:` is sufficient because
                # processing elements by index j ensures the first time we hit a multiple
                # for any j, that j will be the smallest index found *so far* for that multiple value.
                # If we've already assigned an index k (min_index_divisor[multiple] = k),
                # and the current index j > k, we don't update.
                # If we've already assigned an index k (min_index_divisor[multiple] = k),
                # and the current index j < k (this shouldn't happen with the outer loop structure),
                # we would update. But processing j from 0 ensures this simpler logic works.
                
                if min_index_divisor[multiple] == -1:
                     min_index_divisor[multiple] = j
                # If we want to be strictly robust to element order (though not needed here),
                # we would use:
                # if min_index_divisor[multiple] == -1 or j < min_index_divisor[multiple]:
                #     min_index_divisor[multiple] = j

                # Move to the next multiple
                multiple += element_val

        # Now, iterate through the groups and find the assigned element index
        assigned = []
        for group_val in groups:
            # group_val is guaranteed to be between 1 and 10^5 based on constraints,
            # so it's a valid index for our precomputed min_index_divisor array.
            assigned_element_index = min_index_divisor[group_val]
            assigned.append(assigned_element_index)

        return assigned