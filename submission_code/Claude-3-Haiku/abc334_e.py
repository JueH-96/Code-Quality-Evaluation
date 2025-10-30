from collections import deque

def bfs(grid, i, j):
    queue = deque([(i, j)])
    grid[i][j] = '.'
    count = 1
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == '#':
                grid[nx][ny] = '.'
                queue.append((nx, ny))
                count += 1
    return count

def solve():
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    
    green_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                green_count += bfs(grid, i, j)
    
    expected = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                new_grid = [row[:] for row in grid]
                new_grid[i][j] = '#'
                new_green_count = 0
                for x in range(H):
                    for y in range(W):
                        if new_grid[x][y] == '#':
                            new_green_count += bfs(new_grid, x, y)
                expected += (green_count - new_green_count) / (H * W)
    
    return int(expected % 998244353)

print(solve())