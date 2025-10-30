from typing import List
import math

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Step 1: Create a map from element value to its smallest index
        # This map stores the first occurrence index for each unique element value.
        # This allows us to quickly look up the smallest index for any potential divisor value.
        # Time complexity: O(len(elements))
        # Space complexity: O(unique element values), which is at most O(min(len(elements), max_element_value))
        first_occurrence = {}
        for j in range(len(elements)):
            element_value = elements[j]
            # Store the index only if this is the first time we see this value
            if element_value not in first_occurrence:
                first_occurrence[element_value] = j

        # Step 2: Initialize the result array.
        # assigned[i] will store the index of the chosen element for groups[i], or -1.
        # Initialize all entries to -1, which is the default if no suitable element is found.
        # Time complexity: O(len(groups)) for initialization
        # Space complexity: O(len(groups))
        assigned = [-1] * len(groups)

        # Step 3: Process each group to find a suitable element.
        # Iterate through each group. Time complexity: O(len(groups)) iterations.
        for i in range(len(groups)):
            group_size = groups[i]
            
            # Initialize min_element_index to infinity.
            # We want to find the element with the smallest index j that divides group_size.
            # We use infinity as a sentinel value; if it remains infinity after the search,
            # it means no suitable element was found.
            min_element_index = float('inf') 

            # Find divisors of group_size.
            # We can find all divisors by iterating from 1 up to the square root of group_size.
            # If 'k' is a divisor (group_size % k == 0), then 'group_size // k' is also a divisor.
            # We check both 'k' and 'group_size // k'.
            # This loop iterates approximately sqrt(group_size) times.
            
            # Calculate the upper limit for the loop (inclusive).
            # int(math.sqrt(group_size)) gives the integer part of the square root.
            # We add 1 to the limit to ensure the loop includes the integer square root itself
            # in the check when group_size is a perfect square.
            limit = int(math.sqrt(group_size))

            # Iterate from 1 up to the calculated limit.
            # Time complexity for this inner loop: O(sqrt(group_size))
            for k in range(1, limit + 1): 
                # Check if k divides group_size
                if group_size % k == 0:
                    # k is a divisor
                    d1 = k
                    # group_size // k is the corresponding divisor
                    d2 = group_size // k

                    # Check if the divisor value d1 is present in our elements list
                    # by looking it up in the first_occurrence map.
                    if d1 in first_occurrence:
                        # If d1 is in elements, get its smallest index and update
                        # the overall minimum index found for this group so far.
                        min_element_index = min(min_element_index, first_occurrence[d1])

                    # Check the paired divisor value d2.
                    # We only need to check d2 if it's different from d1.
                    # This avoids redundant checks when group_size is a perfect square (k == group_size // k).
                    if d1 != d2 and d2 in first_occurrence:
                        # If d2 is in elements, get its smallest index and update
                        # the overall minimum index found for this group so far.
                        min_element_index = min(min_element_index, first_occurrence[d2])

            # After checking all relevant divisors, if min_element_index was updated from infinity,
            # it means at least one suitable element was found. Its value is the smallest index
            # among all suitable elements.
            # If min_element_index is still float('inf'), it means no suitable element
            # was found that divides group_size, and assigned[i] remains -1 (its initial value).
            if min_element_index != float('inf'):
                 assigned[i] = min_element_index

        # Return the resulting assignment array.
        # Overall time complexity: O(len(elements) + len(groups) * sqrt(max_group_size))
        # Overall space complexity: O(len(elements) + len(groups))
        return assigned