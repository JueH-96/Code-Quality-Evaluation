from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        # We want to count all submatrices whose top-left corner is (0,0)
        # such that in that rectangle #X == #Y and #X >= 1.
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0

        # prev_delta[j] will store the sum over the rectangle (0,0) to (i-1,j)
        # of (X_count - Y_count).  Similarly prev_x[j] for the X count.
        prev_delta = [0] * m
        prev_x     = [0] * m

        answer = 0

        for i in range(n):
            row_delta = 0  # running (X - Y) sum for row i up to column j
            row_x     = 0  # running X count for row i up to column j

            for j in range(m):
                c = grid[i][j]
                if c == 'X':
                    d = 1
                    x = 1
                elif c == 'Y':
                    d = -1
                    x = 0
                else:  # c == '.'
                    d = 0
                    x = 0

                row_delta += d
                row_x     += x

                # total delta and total X in rectangle (0,0) to (i,j):
                curr_delta = prev_delta[j] + row_delta
                curr_x     = prev_x[j]     + row_x

                # check conditions
                if curr_delta == 0 and curr_x >= 1:
                    answer += 1

                # update prefix arrays for the next row
                prev_delta[j] = curr_delta
                prev_x[j]     = curr_x

        return answer