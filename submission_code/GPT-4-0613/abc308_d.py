from collections import deque

H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]
visited = [[False]*W for _ in range(H)]
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
snuke = 'snuke'

def bfs(x, y):
    queue = deque([(x, y, 0)])
    visited[x][y] = True
    while queue:
        x, y, t = queue.popleft()
        if x == H-1 and y == W-1:
            return True
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if nx >= 0 and nx < H and ny >= 0 and ny < W and not visited[nx][ny] and grid[nx][ny] == snuke[(t+1)%5]:
                queue.append((nx, ny, t+1))
                visited[nx][ny] = True
    return False

print('Yes' if bfs(0, 0) else 'No')