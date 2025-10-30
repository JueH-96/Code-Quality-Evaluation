import sys

def read_input():
    H, W, N = map(int, sys.stdin.readline().split())
    T = sys.stdin.readline().strip()
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    return H, W, N, T, grid

def solve(H, W, N, T, grid):
    directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    current = (0, 0)
    visited = set()
    for direction in T:
        dx, dy = directions[direction]
        nx, ny = current[0] + dx, current[1] + dy
        if 0 < nx < H-1 and 0 < ny < W-1 and grid[nx][ny] == '.':
            current = (nx, ny)
            visited.add(current)
    return len(visited)

H, W, N, T, grid = read_input()
print(solve(H, W, N, T, grid))