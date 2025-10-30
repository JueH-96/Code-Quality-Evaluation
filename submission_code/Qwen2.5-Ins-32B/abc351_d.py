# YOUR CODE HERE
import sys
from collections import deque

def bfs(grid, start, visited):
    h, w = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque([start])
    visited[start[0]][start[1]] = True
    count = 1
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == '.':
                visited[nx][ny] = True
                queue.append((nx, ny))
                count += 1
    return count

def solve():
    h, w = map(int, input().split())
    grid = [input().strip() for _ in range(h)]
    max_freedom = 0
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                visited = [[False] * w for _ in range(h)]
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == '#':
                        visited[i][j] = True
                        break
                if not visited[i][j]:
                    max_freedom = max(max_freedom, bfs(grid, (i, j), visited))
    
    print(max_freedom)

solve()