class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Start by assuming the first team (0) is the champion
        champion = 0
        
        # Check each team to see if it can be the champion
        for i in range(1, n):
            # If the current champion is weaker than team i, update the champion
            if grid[champion][i] == 0:
                champion = i
        
        # Verify that the selected champion is indeed stronger than all other teams
        for j in range(n):
            if j != champion and grid[champion][j] == 0:
                # If there is any team j that is stronger than the current champion, return -1
                return -1
        
        return champion