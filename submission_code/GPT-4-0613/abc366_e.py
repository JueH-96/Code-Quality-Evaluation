import sys
from collections import deque

def bfs(sx, sy, ex, ey, d, points):
    visited = [[False]*(2*d+1) for _ in range(2*d+1)]
    q = deque([(sx, sy)])
    visited[sx][sy] = True
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    while q:
        x, y = q.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx <= ex and 0 <= ny <= ey and not visited[nx][ny] and sum(abs(nx-i)+abs(ny-j) for i, j in points) <= d:
                visited[nx][ny] = True
                q.append((nx, ny))
    return sum(visited[i][j] for i in range(2*d+1) for j in range(2*d+1))

def main():
    n, d = map(int, sys.stdin.readline().split())
    points = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    points = [(x-min_x+d, y-min_y+d) for x, y in points]
    print(bfs(0, 0, 2*d, 2*d, d, points))

if __name__ == "__main__":
    main()