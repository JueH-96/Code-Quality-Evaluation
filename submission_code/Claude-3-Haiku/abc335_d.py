import math

def solve():
    N = int(input())
    
    # Create the grid
    grid = [[0] * N for _ in range(N)]
    
    # Place Takahashi at the center
    grid[N//2][N//2] = 'T'
    
    # Place the dragon parts
    part = 1
    for i in range(N):
        for j in range(N):
            if i == N//2 and j == N//2:
                continue
            grid[i][j] = str(part)
            part += 1
    
    # Print the grid
    for row in grid:
        print(' '.join(row))

solve()