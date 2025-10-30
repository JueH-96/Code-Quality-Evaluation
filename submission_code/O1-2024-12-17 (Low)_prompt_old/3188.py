class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Start by assuming team 0 is the champion candidate
        champion = 0
        
        # If a team i is stronger than the current champion, update champion to i
        for i in range(1, n):
            if grid[i][champion] == 1:
                champion = i
        
        # By the problem's transitive guarantee, 'champion' is the actual champion
        return champion