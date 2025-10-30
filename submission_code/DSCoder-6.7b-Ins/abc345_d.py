import sys
from itertools import product

def solve(H, W, N, tiles):
    for configuration in product(*([[(0, 0), (0, 1), (1, 0), (1, 1)]]*N)):
        tiles_positions = [tuple(map(sum, zip(c, t))) for c, t in zip(configuration, tiles)]
        if all(0 <= x < W and 0 <= y < H for x, y in tiles_positions):
            covered = [[False]*W for _ in range(H)]
            for x, y in tiles_positions:
                for dx, dy in product([0, 1], repeat=2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        covered[ny][nx] = True
            if all(all(row) for row in covered):
                return True
    return False

N, H, W = map(int, sys.stdin.readline().split())
tiles = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
print('Yes' if solve(H, W, N, tiles) else 'No')