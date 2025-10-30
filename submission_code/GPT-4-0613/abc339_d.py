from collections import deque
from sys import stdin

def bfs(s, e):
    visited = [[False]*N for _ in range(N)]
    visited[s[0]][s[1]] = True
    queue = deque([s])
    while queue:
        x, y = queue.popleft()
        if [x, y] == e:
            return grid[x][y]
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= N or ny < 0 or ny >= N or visited[nx][ny] or grid[nx][ny] == '#':
                continue
            queue.append([nx, ny])
            visited[nx][ny] = True
            grid[nx][ny] = grid[x][y] + 1
    return -1

N = int(stdin.readline())
grid = [list(stdin.readline().strip()) for _ in range(N)]
players = [[i, j] for i in range(N) for j in range(N) if grid[i][j] == 'P']
print(bfs(players[0], players[1]))