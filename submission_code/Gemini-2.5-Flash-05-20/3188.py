from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)

        # Iterate through each team 'i' to check if it's the champion.
        for i in range(n):
            is_champion = True
            
            # Check if team 'i' is stronger than all other teams 'j'.
            for j in range(n):
                # A team cannot be stronger than itself, skip self-comparison.
                # Also, grid[i][i] is guaranteed to be 0.
                if i == j:
                    continue
                
                # If grid[i][j] is 0, it means team 'j' is stronger than team 'i'.
                # In this case, team 'i' cannot be the champion.
                if grid[i][j] == 0:
                    is_champion = False
                    break # No need to check further for this team 'i'.
            
            # If after checking all other teams, 'is_champion' is still True,
            # then team 'i' is the champion.
            if is_champion:
                return i
        
        # According to the problem constraints and properties (transitivity, total ordering),
        # a unique champion is guaranteed to exist and will be found by the loop.
        # This line should theoretically not be reached.
        return -1 # Sentinel value, indicates an error or unhandled case