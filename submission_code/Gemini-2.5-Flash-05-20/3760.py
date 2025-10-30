import math
from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Step 1: Pre-process elements to store the smallest index for each unique element value.
        # Max value for elements[j] and groups[i] is 10^5.
        MAX_VALUE_POSSIBLE = 100000 
        
        # min_index_for_value[v] will store the smallest index `j` such that elements[j] == v.
        # Initialize with float('inf') to easily find the minimum.
        # The array size is MAX_VALUE_POSSIBLE + 1 because values can be up to MAX_VALUE_POSSIBLE.
        min_index_for_value = [float('inf')] * (MAX_VALUE_POSSIBLE + 1)
        
        for j, el_val in enumerate(elements):
            # As per constraints, 1 <= elements[j] <= 10^5, so el_val will always be <= MAX_VALUE_POSSIBLE.
            # We take the minimum index because if multiple elements have the same value, 
            # we need to pick the one with the smallest index as per problem rules.
            min_index_for_value[el_val] = min(min_index_for_value[el_val], j)
        
        # Step 2: Iterate through groups and find the best assignable element for each.
        assigned = []
        for group_val in groups:
            current_best_j = -1 # Default if no suitable element is found for this group
            min_found_index = float('inf') # Tracks the smallest index `j` found so far for the current group_val
            
            # Find divisors of group_val. Iterate up to sqrt(group_val).
            # For each divisor `k`, `group_val // k` is also a divisor.
            # We only need to check divisors that are potential element values, i.e., <= MAX_VALUE_POSSIBLE.
            # Since elements[j] >= 1, we start k from 1.
            for k in range(1, int(math.sqrt(group_val)) + 1):
                if group_val % k == 0:
                    divisor1 = k
                    divisor2 = group_val // k
                    
                    # Check divisor1
                    # If divisor1 is a value within the range of elements and we have a recorded index for it
                    if divisor1 <= MAX_VALUE_POSSIBLE and min_index_for_value[divisor1] != float('inf'):
                        if min_index_for_value[divisor1] < min_found_index:
                            min_found_index = min_index_for_value[divisor1]
                            current_best_j = min_index_for_value[divisor1]
                            
                    # Check divisor2 (only if it's different from divisor1 to avoid redundant checks for perfect squares)
                    # If divisor2 is a value within the range of elements and we have a recorded index for it
                    if divisor1 != divisor2 and divisor2 <= MAX_VALUE_POSSIBLE and min_index_for_value[divisor2] != float('inf'):
                        if min_index_for_value[divisor2] < min_found_index:
                            min_found_index = min_index_for_value[divisor2]
                            current_best_j = min_index_for_value[divisor2]
            
            assigned.append(current_best_j)
            
        return assigned