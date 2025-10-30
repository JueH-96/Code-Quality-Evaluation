from collections import deque

def solve():
    H, W, Y = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]

    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]

    def bfs(x, y):
        q = deque([(x, y)])
        visited[x][y] = True
        area = 1
        while q:
            x, y = q.popleft()
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and A[nx][ny] > i:
                    visited[nx][ny] = True
                    q.append((nx, ny))
                    area += 1
        return area

    for i in range(Y):
        visited = [[False]*W for _ in range(H)]
        for x in range(H):
            for y in range(W):
                if A[x][y] <= i and not visited[x][y]:
                    area = bfs(x, y)
                    print(area)

solve()