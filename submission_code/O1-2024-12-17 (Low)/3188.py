class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # Start with a candidate champion as team 0
        champion = 0
        
        # Elimination process: if current champion is weaker than i, update champion to i
        for i in range(1, n):
            if grid[champion][i] == 0:
                champion = i
        
        # Double-check that the candidate is indeed not beaten by anyone
        for i in range(n):
            if i != champion and grid[i][champion] == 1:
                return -1  # this would mean no valid champion, but per the problem statement, it won't happen
        
        return champion