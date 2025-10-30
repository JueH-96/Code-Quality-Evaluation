import collections
from typing import List, Set

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        R = len(grid)
        if R == 0:
            return 0
        # C = len(grid[0]) # C is not strictly needed as a global variable, only for loop bounds

        # memoization_table will store the maximum score achievable
        # from a given row_idx, with a specific set of values already selected.
        # The key for memo is (row_idx, frozenset_of_selected_values).
        memo = {} 

        def solve(row_idx: int, current_selected_values: frozenset[int]) -> int:
            """
            Recursively calculates the maximum score from row_idx onwards,
            given the set of values already selected.
            """
            # Base case: If we have processed all rows, no more score can be added.
            if row_idx == R:
                return 0
            
            # Check if this state has been computed before.
            state = (row_idx, current_selected_values)
            if state in memo:
                return memo[state]
            
            # Initialize max_s. This will hold the maximum score for the current state.
            # Start by considering the option of NOT selecting any cell from the current row.
            # In this path, we simply move to the next row with the same set of selected values.
            max_s = solve(row_idx + 1, current_selected_values)
            
            # Option: Iterate through each cell in the current row to potentially select it.
            for val in grid[row_idx]:
                # Condition 2: The values in the set of selected cells must be unique.
                # Check if the current value 'val' has already been selected in previous rows.
                if val not in current_selected_values:
                    # If 'val' is unique, we consider selecting it.
                    # Create a new frozenset including 'val'. frozenset is immutable and hashable,
                    # suitable for use as a dictionary key for memoization.
                    new_selected_values = current_selected_values | {val}
                    
                    # Recursively get the maximum score from the next row (row_idx + 1)
                    # with the updated set of selected values.
                    # Add the current value 'val' to this recursive result to get the score for this path.
                    current_path_score = val + solve(row_idx + 1, new_selected_values)
                    
                    # Update max_s if this path (selecting 'val' from current row)
                    # yields a higher total score compared to other options (including not picking anything from this row).
                    max_s = max(max_s, current_path_score)
            
            # Store the computed maximum score for the current state in the memoization table
            # before returning, to avoid redundant computations.
            memo[state] = max_s
            return max_s

        # Start the recursion from the first row (index 0) with an empty set of selected values.
        return solve(0, frozenset())