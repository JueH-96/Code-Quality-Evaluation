from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        n = len(groups)
        
        # Initialize the result array with -1
        assigned = [-1] * n
        
        # Constraints state group values and element values are between 1 and 10^5.
        # MAX_VAL_CONSTRAINT will be 100,000.
        MAX_VAL_CONSTRAINT = 100000 

        # Create a map from a group's value to a list of indices of groups having that value.
        # Sized MAX_VAL_CONSTRAINT + 1 to handle values up to MAX_VAL_CONSTRAINT.
        value_to_indices_map = [[] for _ in range(MAX_VAL_CONSTRAINT + 1)]
        for i, g_val in enumerate(groups):
            # It's guaranteed 1 <= g_val <= 10^5.
            value_to_indices_map[g_val].append(i)
        
        # Set to keep track of element values that have already been processed.
        # We only care about the first occurrence (smallest index j) of each element value.
        processed_e_vals = set()

        # Iterate through elements with their original indices
        for j, e_val in enumerate(elements):
            # Per constraints, 1 <= elements[i] <= 10^5, so e_val >= 1.
            
            # If this element value has already been processed (i.e., appeared with an earlier index),
            # skip it. Any group it could be assigned to would have already been considered
            # with that earlier, smaller index.
            if e_val in processed_e_vals:
                continue
            
            processed_e_vals.add(e_val)

            # Iterate through all multiples of e_val. These multiples represent group sizes
            # that e_val can divide.
            # We only need to check multiples up to MAX_VAL_CONSTRAINT, as no group size exceeds this.
            current_multiple = e_val
            while current_multiple <= MAX_VAL_CONSTRAINT:
                # If there are any groups with size current_multiple:
                # (value_to_indices_map[current_multiple] will be non-empty)
                for group_idx in value_to_indices_map[current_multiple]:
                    # If this group has not yet been assigned an element:
                    if assigned[group_idx] == -1:
                        # Assign the current element's index j.
                        # This is guaranteed to be the smallest j for this group because
                        # we iterate j in increasing order and assign only once.
                        assigned[group_idx] = j
                
                # Move to the next multiple of e_val.
                current_multiple += e_val
        
        return assigned