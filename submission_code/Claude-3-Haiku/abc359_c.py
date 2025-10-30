from math import ceil

def min_toll(sx, sy, tx, ty):
    dx, dy = tx - sx, ty - sy
    toll = 0

    # Move horizontally
    if dx > 0:
        toll += dx // 2
    else:
        toll += -dx // 2

    # Move vertically
    if dy > 0:
        toll += dy // 2
    else:
        toll += -dy // 2

    return toll

# Read input from stdin
sx, sy, tx, ty = map(int, input().split())

# Compute and print the minimum toll
print(min_toll(sx, sy, tx, ty))