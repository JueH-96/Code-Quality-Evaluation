import sys
import math

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    T = int(data[index])
    index += 1
    results = []

    for _ in range(T):
        K = int(data[index])
        S_x = int(data[index + 1])
        S_y = int(data[index + 2])
        T_x = int(data[index + 3])
        T_y = int(data[index + 4])
        index += 5

        def get_tile(x, y):
            i = x // K
            j = y // K
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                k = y % K
                return (i, j, k)
            else:
                k = x % K
                return (i, j, k)

        start_tile = get_tile(S_x, S_y)
        end_tile = get_tile(T_x, T_y)

        def tile_distance(tile1, tile2):
            i1, j1, k1 = tile1
            i2, j2, k2 = tile2
            return abs(i1 - i2) + abs(j1 - j2) + (0 if k1 == k2 else 1)

        results.append(tile_distance(start_tile, end_tile))

    sys.stdout.write("
".join(map(str, results)) + "
")

# Read from stdin, solve the problem, and write to stdout
if __name__ == "__main__":
    solve()