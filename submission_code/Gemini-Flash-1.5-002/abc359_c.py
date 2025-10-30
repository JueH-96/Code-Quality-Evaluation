sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

def get_tile(x, y):
    return (x + y) // 2

start_tile = get_tile(sx, sy)
end_tile = get_tile(tx, ty)

print(abs(start_tile - end_tile))