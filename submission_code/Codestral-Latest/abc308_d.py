# YOUR CODE HERE
import sys
from collections import deque

def bfs(grid, H, W):
    target = "snukes"
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque([(0, 0, 0)])  # (i, j, index in target)
    visited = set((0, 0, 0))

    while queue:
        i, j, index = queue.popleft()

        if index == 5:
            return True

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == target[index] and (ni, nj, (index + 1) % 6) not in visited:
                    visited.add((ni, nj, (index + 1) % 6))
                    queue.append((ni, nj, (index + 1) % 6))

    return False

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])

    grid = []
    for i in range(H):
        grid.append(data[i + 2])

    if bfs(grid, H, W):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()