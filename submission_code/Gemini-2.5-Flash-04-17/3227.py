from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        max_val = n * n
        
        # Frequency map initialized to zeros
        # Size max_val + 1 to hold counts for numbers 1 to max_val
        # We use indices 1 to max_val. Index 0 is unused.
        counts = [0] * (max_val + 1)
        
        # Iterate through the grid and count frequencies
        for row in grid:
            for num in row:
                # num is guaranteed to be in the range [1, max_val]
                counts[num] += 1
        
        # Initialize variables to hold the results
        repeating_num = None
        missing_num = None
        
        # Iterate from 1 to max_val (inclusive) to find the repeating and missing numbers
        # The problem guarantees exactly one number appears twice and exactly one is missing
        for i in range(1, max_val + 1):
            if counts[i] == 2:
                repeating_num = i
            elif counts[i] == 0:
                missing_num = i
            
            # Optimization: stop once both are found
            # This is safe because we know they exist and are unique in their properties (count 2 or 0)
            if repeating_num is not None and missing_num is not None:
                break
                
        # The problem asks for [repeating_num, missing_num]
        # The variables are guaranteed to be assigned by the end of the loop
        return [repeating_num, missing_num]