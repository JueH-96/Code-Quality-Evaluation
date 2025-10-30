import sys
from collections import deque

def bfs(grid, start, H, W):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([start])
    visited = set([start])
    count = 0

    while queue:
        count += 1
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited and grid[nx][ny] == '.':
                visited.add((nx, ny))
                queue.append((nx, ny))

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    grid = data[2:]

    max_degree = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                max_degree = max(max_degree, bfs(grid, (i, j), H, W))

    print(max_degree)

if __name__ == "__main__":
    main()