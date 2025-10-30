def solve():
    n, h, w = map(int, input().split())
    tiles = []
    for _ in range(n):
        tiles.append(tuple(map(int, input().split())))

    grid_area = h * w

    def check(arrangement):
        total_area = 0
        for tile_index, rotation in arrangement:
            a, b = tiles[tile_index]
            if rotation:
                total_area += b * a
            else:
                total_area += a * b
        return total_area == grid_area

    def is_valid(arrangement):
        for tile_index, rotation in arrangement:
            a, b = tiles[tile_index]
            if rotation:
                if not (b <= h and a <= w):
                    return False
            else:
                if not (a <= h and b <= w):
                    return False
        return True

    import itertools

    for i in range(1 << n):
        used_tiles = []
        for j in range(n):
            if (i >> j) & 1:
                used_tiles.append(j)

        for arrangement in itertools.product([0, 1], repeat=len(used_tiles)):
            current_arrangement = []
            for k in range(len(used_tiles)):
                current_arrangement.append((used_tiles[k], arrangement[k]))

            if is_valid(current_arrangement) and check(current_arrangement):
                print("Yes")
                return

    print("No")

solve()