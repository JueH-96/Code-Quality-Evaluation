sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

# Convert to tile coordinates
def tile_coords(x, y):
    if (x + y) % 2 == 0:
        return (x // 2, y)
    else:
        return (x, y // 2)

start = tile_coords(sx, sy)
end = tile_coords(tx, ty)

# Manhattan distance in tile coordinate system
result = abs(start[0] - end[0]) + abs(start[1] - end[1])
print(result)