from collections import deque

def bfs(start, end, grid, n):
    visited = set()
    queue = deque([(start, 0)])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (x, y), moves = queue.popleft()
        if (x, y) == end:
            return moves
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] != '#' and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), moves + 1))
    
    return -1

def min_moves_to_meet(grid):
    n = len(grid)
    players = [(i, j) for i in range(n) for j in range(n) if grid[i][j] == 'P']
    
    if len(players) != 2:
        return -1
    
    start, end = players
    return bfs(start, end, grid, n)

# Read input
N = int(input().strip())
grid = [input().strip() for _ in range(N)]

# Calculate and print the answer
print(min_moves_to_meet(grid))