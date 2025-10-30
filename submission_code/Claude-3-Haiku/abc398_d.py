from collections import deque

def solve():
    N, R, C = map(int, input().split())
    S = input()
    
    # Initialize the grid with smoke only at (0, 0)
    grid = [[0] * (2*N+1) for _ in range(2*N+1)]
    grid[N][N] = 1
    
    # Define the movement directions
    directions = {'N': (-1, 0), 'W': (0, -1), 'S': (1, 0), 'E': (0, 1)}
    
    # Simulate the smoke movement
    for t in range(N):
        # Move the smoke
        new_grid = [[0] * (2*N+1) for _ in range(2*N+1)]
        for r in range(2*N+1):
            for c in range(2*N+1):
                if grid[r][c] > 0:
                    dr, dc = directions[S[t]]
                    new_r, new_c = r + dr, c + dc
                    new_grid[new_r][new_c] += grid[r][c]
        grid = new_grid
        
        # Generate new smoke at (0, 0) if needed
        if grid[N][N] == 0:
            grid[N][N] = 1
        
        # Check if smoke exists at (R, C) at time t+0.5
        print('1' if grid[N+R][N+C] > 0 else '0', end='')
    
    print()

solve()