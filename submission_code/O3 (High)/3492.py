from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        """
        Count the number of sub-matrices that
          * start at the top-left corner (0,0) – this is forced because the sub-matrix
            must contain the cell (0,0) and indices cannot be negative;
          * contain an equal number of 'X' and 'Y';
          * contain at least one 'X'.
        """
        n = len(grid)
        m = len(grid[0])

        # prefix sums for the previous processed row
        prev_x = [0] * (m + 1)   # number of X's in rectangle (0,0) .. (row-1, col)
        prev_y = [0] * (m + 1)   # number of Y's in rectangle (0,0) .. (row-1, col)

        answer = 0

        # process row by row, keeping only one row of prefix sums at a time
        for r in range(n):
            curr_x = [0] * (m + 1)
            curr_y = [0] * (m + 1)

            row_x = 0         # running number of X's in current row up to column c
            row_y = 0         # running number of Y's in current row up to column c

            for c in range(m):
                ch = grid[r][c]
                if ch == 'X':
                    row_x += 1
                elif ch == 'Y':
                    row_y += 1

                # total X / Y from (0,0) to (r,c) == previous column-wise total
                # for rows above + running total for this row
                curr_x[c + 1] = prev_x[c + 1] + row_x
                curr_y[c + 1] = prev_y[c + 1] + row_y

                # check the three required conditions
                if curr_x[c + 1] > 0 and curr_x[c + 1] == curr_y[c + 1]:
                    answer += 1

            # current row becomes previous row for next iteration
            prev_x, prev_y = curr_x, curr_y

        return answer