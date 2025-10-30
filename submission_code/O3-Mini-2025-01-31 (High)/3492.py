from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        # The problem asks for the number of submatrices that contain grid[0][0]
        # and satisfy:
        #   · Equal number of 'X' and 'Y'
        #   · At least one 'X'
        # Notice that since grid[0][0] is the top‐left cell, any submatrix
        # that contains it must have (0,0) as its top‐left corner.
        # Therefore every valid submatrix is uniquely identified by its
        # bottom–right corner (i, j) (with 0 ≤ i < n and 0 ≤ j < m).
        
        # We can compute the counts for the submatrix spanning (0,0) to (i,j)
        # using a 2D prefix sum technique. In particular, let:
        #   PX[i][j] = number of 'X' in grid[0..i][0..j]
        #   PY[i][j] = number of 'Y' in grid[0..i][0..j]
        # and then we check if PX[i][j] == PY[i][j] and PX[i][j] > 0.
        # To avoid using an extra O(n*m) memory for 2D arrays we can compute
        # the prefix sums row by row using two 1D arrays.
        
        if not grid or not grid[0]:
            return 0
        
        n, m = len(grid), len(grid[0])
        count = 0
        
        # prev_x and prev_y hold the prefix sum values for the previous row.
        prev_x = [0] * m
        prev_y = [0] * m
        
        # Each submatrix is defined by the top-left fixed at (0,0) and 
        # the bottom–right at (i, j). We iterate over i and j computing
        # the prefix sum of 'X' and 'Y' in that submatrix.
        for i in range(n):
            cur_x = [0] * m
            cur_y = [0] * m
            for j in range(m):
                # A cell contributes 1 if it has the appropriate character.
                add_x = 1 if grid[i][j] == 'X' else 0
                add_y = 1 if grid[i][j] == 'Y' else 0
                
                # Start with the current cell's contributions.
                cur_x[j] = add_x
                cur_y[j] = add_y
                
                # Add the prefix from the row above (if any).
                if i > 0:
                    cur_x[j] += prev_x[j]
                    cur_y[j] += prev_y[j]
                
                # Add the prefix for the current row (to the left).
                if j > 0:
                    cur_x[j] += cur_x[j-1]
                    cur_y[j] += cur_y[j-1]
                
                # Remove the double-counted corner (if both above and left exist).
                if i > 0 and j > 0:
                    cur_x[j] -= prev_x[j-1]
                    cur_y[j] -= prev_y[j-1]
                
                # For the submatrix from (0,0) to (i,j) we have:
                #   total_X = cur_x[j] and total_Y = cur_y[j]
                # We count it if:
                #   (1) total_X == total_Y and (2) total_X > 0
                if cur_x[j] == cur_y[j] and cur_x[j] > 0:
                    count += 1
            
            # Prepare for the next row: current row becomes the previous row.
            prev_x, prev_y = cur_x, cur_y
        
        return count

# The code below is useful for local testing and when running the solution as a script.
# It reads input from standard input, expecting first two numbers to be rows and columns,
# then the grid characters.
def solve():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    R = int(next(it))
    C = int(next(it))
    grid = []
    for _ in range(R):
        row = []
        for _ in range(C):
            row.append(next(it))
        grid.append(row)
    
    sol = Solution()
    ans = sol.numberOfSubmatrices(grid)
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    solve()