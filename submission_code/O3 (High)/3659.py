from typing import List

MOD = 10 ** 9 + 7            # required modulus
MAX_XOR = 16                 # every number as well as k is smaller than 16


class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        """
        Dynamic programming.
        dp[c][x] – number of ways to reach the current row, column c
        with an accumulated xor equal to x.
        We sweep the grid row by row keeping only the current row in memory.
        """
        if not grid:
            return 0

        m, n = len(grid), len(grid[0])

        # dp[j] is a length-16 array for column j of the *previously processed* row
        dp = [[0] * MAX_XOR for _ in range(n)]

        # (0, 0) starting point
        dp[0][grid[0][0]] = 1

        for i in range(m):
            # cur_left keeps the vector for the immediate left neighbour in the current row
            cur_left = [0] * MAX_XOR

            for j in range(n):
                cell_val = grid[i][j]

                # the very first cell is already initialised
                if i == 0 and j == 0:
                    cur_left = dp[0]
                    continue

                new_vec = [0] * MAX_XOR

                # paths coming from the cell above (i-1, j)
                if i > 0:
                    for x in range(MAX_XOR):
                        cnt = dp[j][x]
                        if cnt:
                            new_vec[x ^ cell_val] = (new_vec[x ^ cell_val] + cnt) % MOD

                # paths coming from the cell to the left (i, j-1)
                for x in range(MAX_XOR):
                    cnt = cur_left[x]
                    if cnt:
                        new_vec[x ^ cell_val] = (new_vec[x ^ cell_val] + cnt) % MOD

                # store for further processing
                dp[j] = new_vec
                cur_left = new_vec          # becomes the "left neighbour" for next column

        return dp[-1][k] % MOD