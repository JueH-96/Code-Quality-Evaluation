class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])
        val = [[0]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'X':
                    val[i][j] = 1
                elif grid[i][j] == 'Y':
                    val[i][j] = -1
                else:
                    val[i][j] = 0
        count = 0
        for r1 in range(n):
            if r1 > 0:
                continue  # We need to include row 0 (since grid[0][0] must be included)
            sum_col = [0]*m
            for r2 in range(r1, n):
                # Update column sums
                for c in range(m):
                    sum_col[c] += val[r2][c]
                # Now, for this sum_col, process subarrays that start at column 0 (since grid[0][0] must be included)
                cum_sum = 0
                count_map = {0: 1}
                x_count = [0]*m
                x_cum = 0
                for c in range(m):
                    # Update cumulative sum
                    cum_sum += sum_col[c]
                    # Update x_count
                    if grid[r2][c] == 'X':
                        x_cum += 1
                    elif r1 > 0:
                        x_cum += 0
                    else:
                        x_cum += 0
                    # Check if cumulative sum has been seen before
                    if c >= 0:
                        prev_count = count_map.get(cum_sum, 0)
                        if cum_sum == 0 and x_cum > 0:
                            count += prev_count
                        count_map[cum_sum] = count_map.get(cum_sum, 0) + 1
        # Since grid[0][0] must be included, we may need to adjust count
        return count