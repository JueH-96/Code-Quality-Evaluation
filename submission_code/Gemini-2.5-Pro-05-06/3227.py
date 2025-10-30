from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        # N_squared represents n*n. This is the count of numbers if the set 1...N_squared was complete,
        # and also the maximum value any number in the grid can take.
        N_squared = n * n 
        
        # Create a frequency array to store counts of each number.
        # The numbers are in the range [1, N_squared].
        # We use an array of size N_squared + 1 for 1-based indexing (counts[0] is unused).
        # Initialize all counts to 0.
        counts = [0] * (N_squared + 1) 
        
        # Populate counts from the grid elements.
        # Iterate through each row in the grid.
        for row in grid:
            # Iterate through each number in the current row.
            for num in row:
                # The problem constraints guarantee 1 <= num <= N_squared.
                counts[num] += 1
                
        repeated_num = 0  # Initialize with a value indicating not found (actual numbers are >= 1).
        missing_num = 0   # Initialize with a value indicating not found.
        
        # Find the repeated and missing numbers by checking frequencies.
        # Iterate through all possible numbers from 1 to N_squared.
        for i in range(1, N_squared + 1):
            if counts[i] == 2:
                # This number 'i' appeared twice in the grid.
                repeated_num = i
            elif counts[i] == 0:
                # This number 'i' (from the expected set 1...N_squared) did not appear in the grid.
                missing_num = i
            
            # Optimization: If both repeated_num and missing_num have been found,
            # we can break out of the loop early. This is a minor optimization
            # as N_squared is at most 2500, but good practice.
            if repeated_num != 0 and missing_num != 0:
                break
                
        # The problem guarantees that exactly one number is repeated and one is missing.
        # Thus, repeated_num and missing_num will be updated from their initial 0 values.
        return [repeated_num, missing_num]