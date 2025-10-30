def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())

    def get_tile_id(x, y):
        if (x + y) % 2 == 0:
            return (x // 2, y)
        else:
            return (x // 2, y)

    start_tile = get_tile_id(sx, sy)
    end_tile = get_tile_id(tx, ty)

    print(abs(start_tile[0] - end_tile[0]) + abs(start_tile[1] - end_tile[1]))

solve()