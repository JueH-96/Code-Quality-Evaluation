def solve(N):
    # Initialize grid
    grid = [['' for _ in range(N)] for _ in range(N)]
    
    for i in range(1, N+1):
        j = N + 1 - i
        if i <= j:
            # Color is black ('#') if i is odd, white ('.') if i is even
            color = '#' if i % 2 == 1 else '.'
            
            # Fill the rectangular region from (i,i) to (j,j)
            for r in range(i-1, j):
                for c in range(i-1, j):
                    grid[r][c] = color
    
    # Print the grid
    for row in grid:
        print(''.join(row))

# Read input
N = int(input())
solve(N)