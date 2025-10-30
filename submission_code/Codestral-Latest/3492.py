class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        def count_submatrices_with_equal_X_and_Y(row):
            count = 0
            x_count = 0
            y_count = 0
            for char in row:
                if char == 'X':
                    x_count += 1
                elif char == 'Y':
                    y_count += 1
                if x_count == y_count and x_count > 0:
                    count += 1
            return count

        def count_submatrices_in_range(grid, start_row, end_row):
            count = 0
            for i in range(start_row, end_row + 1):
                for j in range(i, end_row + 1):
                    row = []
                    for k in range(len(grid[0])):
                        if grid[i][k] == 'X' or grid[j][k] == 'X':
                            row.append('X')
                        elif grid[i][k] == 'Y' or grid[j][k] == 'Y':
                            row.append('Y')
                        else:
                            row.append('.')
                    count += count_submatrices_with_equal_X_and_Y(row)
            return count

        total_count = 0
        for i in range(len(grid)):
            for j in range(i, len(grid)):
                total_count += count_submatrices_in_range(grid, i, j)

        return total_count