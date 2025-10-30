class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for a in range(n):
            # Check if all entries in column a (except diagonal) are 0
            for i in range(n):
                if i != a and grid[i][a] == 1:
                    break
            else:
                return a
        return -1  # According to constraints, this line is unreachable