def get_tile_id(x, y):
    if (x + y) % 2 == 0:
        # Horizontal tile
        return (x - x % 2, y)
    else:
        # Vertical tile
        return (x, y - y % 2)

# Read input
sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

# Check if same position
if sx == tx and sy == ty:
    print(0)
    exit()

# Check if in same tile
start_tile = get_tile_id(sx, sy)
end_tile = get_tile_id(tx, ty)

if start_tile == end_tile:
    print(0)
    exit()

# Calculate the minimum number of tiles to cross
dx = abs(tx - sx)
dy = abs(ty - sy)

# The optimal path typically involves diagonal movement when possible
# followed by straight movement along the remaining axis

# When moving diagonally, we cross approximately one tile boundary per step
# When moving straight, we cross fewer boundaries due to the 2x1 tile structure

diagonal_steps = min(dx, dy)
straight_steps = max(dx, dy) - min(dx, dy)

# Calculate total tiles crossed based on the movement pattern
# Diagonal movement: each step typically enters a new tile
# Straight movement: crosses boundaries less frequently (about every 2 steps)
total_tiles = diagonal_steps + (straight_steps + 1) // 2

print(total_tiles)