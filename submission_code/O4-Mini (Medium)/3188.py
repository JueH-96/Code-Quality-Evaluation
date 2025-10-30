from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        """
        We want to find the team a such that no other team is stronger than a.
        Because the "stronger than" relation is complete and transitive across all teams,
        we can find the maximum (champion) in O(n) time by:
        
        1. Initialize candidate = 0
        2. For each i from 1 to n-1:
           - if grid[candidate][i] == 0, then i is stronger than candidate, so set candidate = i
        3. Return the final candidate.
        
        This works like the classic "find the maximum in a tournament" algorithm.
        """

        n = len(grid)
        candidate = 0
        for i in range(1, n):
            # if candidate loses to i, i becomes the new candidate
            if grid[candidate][i] == 0:
                candidate = i
        return candidate