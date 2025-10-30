from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        """
        Finds the champion in a tournament.

        Args:
            grid: A 2D boolean matrix where grid[i][j] == 1 means team i is
                  stronger than team j.

        Returns:
            The index of the champion team.
        """
        n = len(grid)
        
        # Start by assuming team 0 is the champion.
        champion = 0
        
        # Iterate through all other teams to see if any can defeat the current champion.
        for i in range(1, n):
            # If team `i` is stronger than the current `champion`, then team `i`
            # becomes the new champion candidate.
            if grid[i][champion] == 1:
                champion = i
        
        # After one pass, due to the transitivity property, the final candidate
        # is the strongest team and thus the champion.
        return champion