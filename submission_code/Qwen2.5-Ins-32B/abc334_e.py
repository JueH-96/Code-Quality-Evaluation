import sys
from collections import deque

def bfs(grid, visited, start, h, w):
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    q = deque([start])
    visited[start[0]][start[1]] = True
    while q:
        x, y = q.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == '#':
                visited[nx][ny] = True
                q.append((nx, ny))

def count_components(grid, h, w):
    visited = [[False] * w for _ in range(h)]
    count = 0
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] == '#':
                bfs(grid, visited, (i, j), h, w)
                count += 1
    return count

def main():
    input = sys.stdin.read
    data = input().split()
    
    h, w = int(data[0]), int(data[1])
    grid = data[2:2+h]
    
    red_count = sum(row.count('.') for row in grid)
    mod = 998244353
    
    current_components = count_components(grid, h, w)
    
    total_increase = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                new_grid = [row[:] for row in grid]
                new_grid[i] = new_grid[i][:j] + '#' + new_grid[i][j+1:]
                new_components = count_components(new_grid, h, w)
                total_increase += new_components - current_components
    
    expected_value = (current_components * red_count + total_increase) * pow(red_count, mod-2, mod)
    print(expected_value % mod)

main()