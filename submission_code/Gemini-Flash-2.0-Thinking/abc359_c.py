def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())

    def get_tile_coords(x, y):
        ix = int(x)
        iy = int(y)
        return ix, iy

    def get_tile_index(ix, iy):
        if (ix + iy) % 2 == 0:
            return (ix // 2, iy)
        else:
            return (ix, iy // 2)

    start_ix, start_iy = get_tile_coords(sx, sy)
    target_ix, target_iy = get_tile_coords(tx, ty)

    start_tile_index = get_tile_index(start_ix, start_iy)
    target_tile_index = get_tile_index(target_ix, target_iy)

    if start_tile_index == target_tile_index:
        print(0)
        return

    print(abs(start_tile_index[0] - target_tile_index[0]) + abs(start_tile_index[1] - target_tile_index[1]))

solve()