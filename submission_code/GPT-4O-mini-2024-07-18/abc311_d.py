def count_ice_squares(N, M, grid):
    # Directions for movement: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # To keep track of visited squares
    visited = [[False] * M for _ in range(N)]
    
    # Starting position
    start_x, start_y = 1, 1  # (2, 2) in 1-based index is (1, 1) in 0-based index
    visited[start_x][start_y] = True
    
    # Function to perform the movement in a given direction
    def move(x, y, dx, dy):
        while 0 <= x < N and 0 <= y < M and grid[x][y] == '.':
            if not visited[x][y]:
                visited[x][y] = True
            x += dx
            y += dy
        return x - dx, y - dy  # Return the last valid position
    
    # Explore all directions from the starting position
    for dx, dy in directions:
        move(start_x, start_y, dx, dy)
    
    # Count the number of ice squares visited
    ice_count = sum(visited[i][j] for i in range(N) for j in range(M) if grid[i][j] == '.')
    
    return ice_count

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
grid = data[1:N+1]

# Get the result and print it
result = count_ice_squares(N, M, grid)
print(result)