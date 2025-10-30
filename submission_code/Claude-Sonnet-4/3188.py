class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        for team in range(n):
            # Count how many teams this team beats
            wins = sum(grid[team])
            
            # If this team beats all other teams, it's the champion
            if wins == n - 1:
                return team
        
        return -1