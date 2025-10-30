class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        in_degree = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and grid[j][i] == 1:
                    in_degree[i] += 1
        for i in range(n):
            if in_degree[i] == 0:
                return i
        return -1  # Should not happen if input is valid