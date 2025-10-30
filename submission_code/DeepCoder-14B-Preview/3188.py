class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for a in range(n):
            is_champ = True
            for j in range(n):
                if a == j:
                    continue
                if grid[a][j] != 1:
                    is_champ = False
                    break
            if is_champ:
                return a
        return -1  # This line is theoretically unreachable due to problem constraints