class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Count how many teams are stronger than each team
        losses = [0] * n
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    # Team i is stronger than team j
                    # So team j has one more loss
                    losses[j] += 1
        
        # The champion is the team with 0 losses
        for team in range(n):
            if losses[team] == 0:
                return team
        
        # Should never reach here based on problem constraints
        return -1