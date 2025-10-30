import sys

def solve():
    N = int(sys.stdin.readline())

    grid = [['' for _ in range(N)] for _ in range(N)]

    for r0 in range(N): # 0-indexed row
        for c0 in range(N): # 0-indexed column
            # Calculate the "depth" d of the cell (r0, c0).
            # Depth is the minimum distance from the cell to any of the four outer boundaries of the grid.
            # For a cell (r0, c0) in an N x N grid (0-indexed):
            # Distance to top boundary: r0
            # Distance to left boundary: c0
            # Distance to bottom boundary: (N-1) - r0
            # Distance to right boundary: (N-1) - c0
            # So, d = min(r0, c0, (N-1)-r0, (N-1)-c0).
            # This d is a 0-indexed frame number. d=0 is outermost, d=1 is next, etc.
            d = min(r0, c0, (N - 1) - r0, (N - 1) - c0)
            
            # The problem states operations are indexed by 'i' (1-indexed loop variable).
            # The cell (r0, c0) with depth d is last affected by the operation where
            # the 1-indexed loop variable 'i' is equal to d+1.
            # Let's call this the effective operation index.
            effective_op_idx = d + 1
            
            # The color rule is: black ('#') if the effective operation index is odd, 
            # and white ('.') if it's even.
            if effective_op_idx % 2 == 1:  # If d+1 is odd (i.e., d is even)
                grid[r0][c0] = '#'
            else:  # If d+1 is even (i.e., d is odd)
                grid[r0][c0] = '.'

    # Print the resulting grid.
    for i in range(N):
        print("".join(grid[i]))

if __name__ == '__main__':
    solve()