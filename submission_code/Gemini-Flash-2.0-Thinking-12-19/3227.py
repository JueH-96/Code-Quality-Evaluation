class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        counts = [0] * (n * n + 1)
        for row in grid:
            for val in row:
                counts[val] += 1

        repeated = -1
        missing = -1
        for i in range(1, n * n + 1):
            if counts[i] == 2:
                repeated = i
            if counts[i] == 0:
                missing = i
        return [repeated, missing]