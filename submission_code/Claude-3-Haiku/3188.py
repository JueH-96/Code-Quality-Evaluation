class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Create a dictionary to store the number of teams that are stronger than each team
        stronger_count = {i: 0 for i in range(n)}
        
        # Count the number of teams that are stronger than each team
        for i in range(n):
            for j in range(n):
                if i != j and grid[i][j] == 1:
                    stronger_count[j] += 1
        
        # Find the team with no team stronger than it
        for i in range(n):
            if stronger_count[i] == 0:
                return i
        
        # If no such team is found, return -1
        return -1