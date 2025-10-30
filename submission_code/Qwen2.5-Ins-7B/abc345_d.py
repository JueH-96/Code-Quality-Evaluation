# YOUR CODE HERE
from itertools import product

def can_place_tiles(N, H, W, tiles):
    def is_valid(x, y, tile):
        a, b = tile
        if a > b:
            a, b = b, a
        for dx, dy in product([-1, 0, 1], repeat=2):
            if dx == 0 and dy == 0:
                continue
            for i in range(a):
                for j in range(b):
                    nx, ny = x + i + dx, y + j + dy
                    if 0 <= nx < H and 0 <= ny < W and (nx, ny) in placed:
                        return False
                    placed.add((nx, ny))
        return True

    def backtrack(index):
        if index == N:
            return True
        for x in range(H):
            for y in range(W):
                if (x, y) not in placed:
                    for tile in tiles:
                        if is_valid(x, y, tile):
                            placed.add((x, y))
                            if backtrack(index + 1):
                                return True
                            placed.remove((x, y))
                    break
        return False

    placed = set()
    return backtrack(0)

N, H, W = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(N)]
if can_place_tiles(N, H, W, tiles):
    print("Yes")
else:
    print("No")