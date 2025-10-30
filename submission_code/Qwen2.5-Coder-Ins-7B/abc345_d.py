import sys
from collections import defaultdict

def can_place_tiles(N, H, W, tiles):
    def is_valid(x, y, a, b):
        return 0 <= x <= H - a and 0 <= y <= W - b

    def place_tile(x, y, a, b):
        for i in range(a):
            for j in range(b):
                grid[x + i][y + j] = 1

    def remove_tile(x, y, a, b):
        for i in range(a):
            for j in range(b):
                grid[x + i][y + j] = 0

    def backtrack(i):
        if i == N:
            return True
        a, b = tiles[i]
        for x in range(H):
            for y in range(W):
                if is_valid(x, y, a, b):
                    place_tile(x, y, a, b)
                    if backtrack(i + 1):
                        return True
                    remove_tile(x, y, a, b)
                if is_valid(x, y, b, a):
                    place_tile(x, y, b, a)
                    if backtrack(i + 1):
                        return True
                    remove_tile(x, y, b, a)
        return False

    grid = [[0] * W for _ in range(H)]
    return backtrack(0)

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = int(data[1])
    W = int(data[2])
    tiles = [(int(data[3 + 2 * i]), int(data[4 + 2 * i])) for i in range(N)]
    if can_place_tiles(N, H, W, tiles):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()