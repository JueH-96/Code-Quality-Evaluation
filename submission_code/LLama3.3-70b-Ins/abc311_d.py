from collections import deque

def solve():
    n, m = map(int, input().split())
    grid = [list(input()) for _ in range(n)]

    visited = [[False]*m for _ in range(n)]
    queue = deque([(1, 1)])
    visited[1][1] = True
    count = 1

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x, y
            while 0 < nx < n-1 and 0 < ny < m-1 and grid[nx][ny] == '.':
                nx += dx
                ny += dy
            if 0 < nx < n-1 and 0 < ny < m-1 and not visited[nx][ny]:
                queue.append((nx, ny))
                visited[nx][ny] = True
                count += 1

    print(count)

if __name__ == "__main__":
    solve()