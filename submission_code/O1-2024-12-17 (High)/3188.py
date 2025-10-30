class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # Start with 0 as a potential champion
        candidate = 0
        # Find the potential champion
        for i in range(1, n):
            if grid[i][candidate] == 1:
                candidate = i
        
        # Verify that the candidate is not weaker than any other team
        for j in range(n):
            if j != candidate and grid[j][candidate] == 1:
                # In theory this should not happen given the problem constraints,
                # but we include the check for completeness.
                return -1
        
        return candidate