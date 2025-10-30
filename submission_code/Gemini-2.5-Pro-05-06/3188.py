from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Iterate through each team `i`. Team `i` is a candidate for champion.
        for i in range(n):
            # Assume team `i` is the champion until proven otherwise.
            # A team `i` is the champion if no other team `j` is stronger than `i`.
            # Team `j` is stronger than team `i` if grid[j][i] == 1.
            
            # This flag tracks if team `i` is indeed a champion.
            # It remains true if no other team is found to be stronger than `i`.
            is_i_the_champion = True 
            
            # Check against all other teams `j`.
            for j in range(n):
                if i == j:
                    # A team is not compared against itself.
                    # The problem defines "stronger" for i != j.
                    # Also, grid[i][i] is 0 by constraint, so grid[i][i] == 1 would be false anyway.
                    continue
                
                if grid[j][i] == 1:
                    # Team `j` is stronger than team `i`.
                    # Therefore, team `i` cannot be the champion.
                    is_i_the_champion = False
                    # No need to check other opponents for team `i`;
                    # we already found one that defeats `i`.
                    break 
            
            if is_i_the_champion:
                # If the inner loop completed and `is_i_the_champion` is still True,
                # it means no other team `j` was found to be stronger than team `i`.
                # So, team `i` is the champion.
                return i
        
        # This part of the code should be unreachable given the problem constraints.
        # The problem guarantees that a unique champion exists due to the properties
        # of the grid (transitivity, grid[i][j] != grid[j][i], etc.).
        # This fallback return is for defensive programming or to satisfy static analyzers
        # that might not infer that the loop always returns a value.
        return -1 # Should not be reached in a valid test case as per problem statement.