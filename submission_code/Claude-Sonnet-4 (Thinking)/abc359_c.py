def get_tile(i, j):
    if (i + j) % 2 == 0:
        return (i, j)
    else:
        return (i - 1, j)

def min_toll(sx, sy, tx, ty):
    start_tile = get_tile(sx, sy)
    target_tile = get_tile(tx, ty)
    
    a1, b1 = start_tile
    a2, b2 = target_tile
    
    distance = max(
        (abs(a1 - a2) + 1) // 2,
        abs(b1 - b2),
        (abs(a1 - a2 + b1 - b2) + 1) // 2
    )
    
    return distance

# Read input
sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

# Output result
print(min_toll(sx, sy, tx, ty))