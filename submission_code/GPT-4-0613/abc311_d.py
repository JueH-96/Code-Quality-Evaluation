from collections import deque

def solve():
    n, m = map(int, input().split())
    grid = [list(input()) for _ in range(n)]
    visited = [[False]*m for _ in range(n)]
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    queue = deque([(1, 1)])
    visited[1][1] = True
    count = 0
    while queue:
        x, y = queue.popleft()
        count += 1
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            while 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == '.' and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append((nx, ny))
                nx += dx[i]
                ny += dy[i]
    print(count)

solve()