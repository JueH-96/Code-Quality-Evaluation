from collections import deque

def is_valid(grid):
    visited = [[False]*4 for _ in range(4)]
    dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '#' and not visited[i][j]:
                queue = deque([(i, j)])
                visited[i][j] = True
                while queue:
                    x, y = queue.popleft()
                    for k in range(4):
                        nx, ny = x + dx[k], y + dy[k]
                        if 0 <= nx < 4 and 0 <= ny < 4 and grid[nx][ny] == '#' and not visited[nx][ny]:
                            queue.append((nx, ny))
                            visited[nx][ny] = True
                return False
    return True

def solve():
    grid = [['.']*4 for _ in range(4)]
    for i in range(12):
        x, y, c = input().split()
        x, y = int(x) - 1, int(y) - 1
        grid[x][y] = c
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '#':
                for k in range(4):
                    nx, ny = i + dx[k], j + dy[k]
                    if 0 <= nx < 4 and 0 <= ny < 4 and grid[nx][ny] == '#':
                        grid[i][j] = '.'
                        if is_valid(grid):
                            print('Yes')
                            return
                        grid[i][j] = '#'
    print('No')

solve()