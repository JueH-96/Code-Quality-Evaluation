class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        counts = {}
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                counts[num] = counts.get(num, 0) + 1

        repeated = -1
        missing = -1

        for i in range(1, n * n + 1):
            count = counts.get(i, 0)
            if count == 2:
                repeated = i
            elif count == 0:
                missing = i

        return [repeated, missing]