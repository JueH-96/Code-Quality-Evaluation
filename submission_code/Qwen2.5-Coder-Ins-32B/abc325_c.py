# YOUR CODE HERE
import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    grid = input[2:]

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def bfs(x, y):
        queue = deque([(x, y)])
        grid[x] = grid[x][:y] + '.' + grid[x][y+1:]
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                    grid[nx] = grid[nx][:ny] + '.' + grid[nx][ny+1:]
                    queue.append((nx, ny))

    sensor_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                bfs(i, j)
                sensor_count += 1

    print(sensor_count)

if __name__ == "__main__":
    main()