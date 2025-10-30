# Read input
sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

# Calculate dx and dy
dx = abs(sx - tx)
dy = abs(sy - ty)

# Check if start and end are in the same tile
start_parity = (sx + sy) % 2
end_parity = (tx + ty) % 2

# Check if they are in the same tile
same_tile = False
if start_parity == end_parity:
    if (sx <= tx <= sx + 2 and sy == ty) or (sy <= ty <= sy + 2 and sx == tx):
        same_tile = True

if same_tile:
    print(0)
else:
    # Calculate Manhattan distance
    manhattan = dx + dy
    # Adjust based on parity
    if (start_parity == end_parity) and (manhattan % 2 == 0):
        print(manhattan // 2)
    else:
        print(manhattan)