from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Initialize a list to store the number of teams each team is stronger than
        stronger_than = [0] * n
        
        # Count the number of teams each team is stronger than
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    stronger_than[i] += 1
        
        # The champion is the team that is stronger than the most teams
        champion = stronger_than.index(max(stronger_than))
        
        # Check if the champion is not stronger than any team that is stronger than it
        for i in range(n):
            if i != champion and grid[i][champion] == 1:
                return -1  # No champion
        
        return champion