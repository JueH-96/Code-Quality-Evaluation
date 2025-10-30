import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    grid = data[2:]

    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W

    def bfs(start_x, start_y):
        queue = deque([(start_x, start_y)])
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and grid[nx][ny] == '#' and not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    visited = [[False] * W for _ in range(H)]
    sensor_count = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                sensor_count += 1
                visited[i][j] = True
                bfs(i, j)

    print(sensor_count)

if __name__ == "__main__":
    main()