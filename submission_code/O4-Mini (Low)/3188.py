from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        """
        We want to find the team 'a' such that no other team b is stronger than a.
        Given grid[i][j] == 1 means i is stronger than j, otherwise j is stronger than i.
        The relation is total and transitive, so there is a single maximum.
        
        We can find the champion in O(n) by a "candidate elimination" approach:
        - Start with candidate = 0.
        - For each other team i:
            - If candidate is not stronger than i (i.e. grid[candidate][i] == 0),
              then i must be stronger than candidate, so update candidate = i.
        - At the end, candidate is the strongest team.
        """
        n = len(grid)
        candidate = 0
        for i in range(1, n):
            # If candidate loses to i, then i is stronger => new candidate
            if grid[candidate][i] == 0:
                candidate = i
        return candidate