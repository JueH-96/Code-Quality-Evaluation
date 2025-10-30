from collections import deque

def max_degree_of_freedom(H, W, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'
    
    def has_adjacent_magnet(x, y):
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                return True
        return False
    
    def bfs(x, y):
        visited = set()
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) not in visited:
                visited.add((cx, cy))
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if is_valid(nx, ny) and not has_adjacent_magnet(nx, ny) and (nx, ny) not in visited:
                        queue.append((nx, ny))
        return len(visited)
    
    max_freedom = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not has_adjacent_magnet(i, j):
                max_freedom = max(max_freedom, bfs(i, j))
    
    return max_freedom

# Read input
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Solve and print the result
print(max_degree_of_freedom(H, W, grid))