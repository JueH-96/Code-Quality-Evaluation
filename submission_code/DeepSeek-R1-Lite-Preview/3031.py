from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        flat_grid = [num for row in grid for num in row]
        zero_count = flat_grid.count(0)
        total_product = 1
        for num in flat_grid:
            if num != 0:
                total_product = (total_product * num) % MOD
            else:
                zero_count += 1  # Redundant but ensures correctness

        result = []
        idx = 0  # To track position in the flat list
        for i in range(n):
            row = []
            for j in range(m):
                current_num = flat_grid[idx]
                if zero_count > 1:
                    row.append(0)
                elif zero_count == 1:
                    if current_num == 0:
                        row.append(total_product)
                    else:
                        row.append(0)
                else:
                    try:
                        inverse = pow(current_num, -1, MOD)
                        value = (total_product * inverse) % MOD
                        row.append(value)
                    except ValueError:
                        row.append(0)
                idx += 1
            result.append(row)
        return result