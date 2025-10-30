import math
from typing import List

class Solution:
    """
    Solves the problem of assigning elements to groups based on divisibility rules.

    The problem asks us to assign an element `elements[j]` to each group `groups[i]` such that
    `groups[i]` is divisible by `elements[j]`. If multiple elements satisfy this condition,
    we must choose the element with the smallest index `j`. If no element satisfies the
    condition for a group, -1 should be assigned.

    The algorithm uses the following approach:

    1. Precomputation:
       - Create a hash map `element_index_map` to store the first occurrence (smallest index)
         of each unique element value found in the `elements` array. This map allows for
         quick O(1) average time lookup of the required index `j` for a given element value.
       - This step takes O(M) time, where M is the number of elements.

    2. Assignment:
       - Iterate through each group `groups[i]` using its index `i`.
       - For each group size `g = groups[i]`:
         - Find all divisors `d` of `g`. An efficient way to do this is to iterate from
           `k = 1` up to `sqrt(g)`. If `k` divides `g`, then both `k` and `g // k` are divisors.
         - For each divisor `d` found:
           - Check if `d` exists as a key in the `element_index_map`. This tells us if an
             element with value `d` is available in the `elements` array.
           - If `d` is in the map, retrieve its associated smallest index `j = element_index_map[d]`.
           - Keep track of the minimum index `min_j` encountered among all valid divisors for
             the current group `g`.
         - After checking all divisors, if at least one valid divisor (and corresponding element)
           was found (`found_divisor` flag is True), assign `assigned[i] = min_j`.
         - If no suitable element was found for group `i`, `assigned[i]` remains at its
           initialized value of -1.
       - This step iterates through N groups. Finding divisors for group `g` takes roughly
         O(sqrt(g)) time. Checking divisors in the map takes O(1) on average.
         The total time for this step is approximately O(N * sqrt(max(groups))).

    The overall time complexity is dominated by the assignment step, resulting in
    O(M + N * sqrt(max_val)), where N = len(groups), M = len(elements), and
    max_val = max(groups).
    The space complexity is O(U + N), where U is the number of unique elements in `elements`
    (for the map) and N is for the output array `assigned`. U is at most M.
    """
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        """
        Assigns elements to groups based on divisibility and smallest index preference.

        Args:
            groups: A list of integers where groups[i] is the size of the i-th group.
            elements: A list of integers representing the available elements.

        Returns:
            An integer array `assigned` where assigned[i] is the index of the element 
            chosen for group i, or -1 if no suitable element exists.
        """
        
        n = len(groups)  # Number of groups
        m = len(elements) # Number of elements
        
        # Initialize the result array with -1 for all groups initially.
        # assigned[i] will store the index 'j' of the assigned element, or -1.
        assigned = [-1] * n 

        # --- Precomputation Step ---
        # Create a map to store the smallest index for each unique element value.
        # Key: element value, Value: smallest index j such that elements[j] == key.
        element_index_map = {}
        # Iterate through the elements array with index j
        for j in range(m):
            e = elements[j]
            # If this element value 'e' is not already in the map, it means this is
            # the first time we encounter it. Record its index 'j' as it's the smallest.
            # Subsequent occurrences of 'e' will have larger indices and are ignored
            # because we only care about the element with the smallest index.
            if e not in element_index_map:
                element_index_map[e] = j

        # --- Assignment Step ---
        # Iterate through each group using its index 'i'
        for i in range(n):
            g = groups[i] # Current group size
            
            # Variables to track the best assignment for this group 'i'
            min_j = float('inf') # Stores the minimum element index found so far, initialized to infinity.
            found_divisor = False # Flag indicating if any assignable element was found for group 'i'.

            # --- Find Divisors and Check Assignment Eligibility ---
            # Efficiently find divisors by iterating from k=1 up to the square root of 'g'.
            # Using integer square root is sufficient.
            limit = int(math.sqrt(g))
            for k in range(1, limit + 1):
                # If 'k' is a divisor of 'g'
                if g % k == 0:
                    # Check divisor 'k': Can an element with value 'k' be assigned?
                    # Check if 'k' exists as a value in the 'elements' array via the map.
                    if k in element_index_map:
                        # Yes, element 'k' exists. Get its smallest index 'j'.
                        j = element_index_map[k]
                        # Update the minimum index if 'j' is smaller than the current minimum.
                        min_j = min(min_j, j)
                        found_divisor = True # Mark that we found at least one valid element.
                    
                    # Check the complementary divisor 'd2 = g // k'.
                    d2 = g // k
                    # Avoid processing the same divisor twice if 'g' is a perfect square (k*k = g).
                    # If k == d2, we already processed it when checking 'k'.
                    if k != d2: 
                        # Check divisor 'd2': Can an element with value 'd2' be assigned?
                        # Check if 'd2' exists as a value in the 'elements' array via the map.
                        if d2 in element_index_map:
                            # Yes, element 'd2' exists. Get its smallest index 'j'.
                            j = element_index_map[d2]
                             # Update the minimum index if 'j' is smaller.
                            min_j = min(min_j, j)
                            found_divisor = True # Mark that we found at least one valid element.
            
            # --- Finalize Assignment for Group 'i' ---
            # After checking all divisors, if we found at least one suitable element:
            if found_divisor:
                # Assign the minimum index found (`min_j`) to the result array for group 'i'.
                # This 'min_j' corresponds to the element with the smallest index that divides 'g'.
                assigned[i] = min_j
            # Otherwise (if no suitable element was found), `assigned[i]` remains -1 (its default value).

        # Return the final assignment array containing the indices of assigned elements or -1.
        return assigned