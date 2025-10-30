class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_sum = n * (n + 1) // 2
        total_square_sum = n * (n + 1) * (2 * n + 1) // 6
        missing = 0
        repeating = 0

        row_sum = 0
        col_sum = 0
        row_square_sum = 0
        col_square_sum = 0

        for i in range(n):
            row_sum += sum(grid[i])
            col_sum += sum([grid[j][i] for j in range(n)])
            row_square_sum += sum([grid[i][j]**2 for j in range(n)])
            col_square_sum += sum([grid[j][i]**2 for j in range(n)])

        s1 = total_sum - row_sum
        s2 = total_sum - col_sum
        p1 = total_square_sum - row_square_sum
        p2 = total_square_sum - col_square_sum

        x = (s1 - s2) // 2
        y = (p1 - p2) // 2

        missing = (x + y) // 2
        repeating = missing - x

        return [repeating, missing]