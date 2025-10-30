def max_degree_of_freedom(H, W, grid):
    from collections import deque

    # Directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # To track visited cells
    visited = [[False] * W for _ in range(H)]
    
    def bfs(start_i, start_j):
        queue = deque([(start_i, start_j)])
        visited[start_i][start_j] = True
        count = 0
        
        while queue:
            i, j = queue.popleft()
            count += 1
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj] and grid[ni][nj] == '.':
                    # Check if adjacent cells have magnets
                    if not any(0 <= ni + ddi < H and 0 <= nj + ddj < W and grid[ni + ddi][nj + ddj] == '#' 
                               for ddi, ddj in directions):
                        visited[ni][nj] = True
                        queue.append((ni, nj))
        
        return count

    max_freedom = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not visited[i][j]:
                # Check if this cell is surrounded by magnets
                if not any(0 <= i + di < H and 0 <= j + dj < W and grid[i + di][j + dj] == '#' 
                           for di, dj in directions):
                    freedom = bfs(i, j)
                    max_freedom = max(max_freedom, freedom)
    
    return max_freedom

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
grid = data[1:H + 1]

# Get the result
result = max_degree_of_freedom(H, W, grid)

# Print the result
print(result)