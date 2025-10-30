def same_tile(i1, j1, i2, j2):
    if (i1 + j1) % 2 == 0:
        # Horizontal tile: same j, i or i+1
        return (i2 == i1 or i2 == i1 + 1) and j2 == j1
    else:
        # Vertical tile: same i, j or j+1
        return (j2 == j1 or j2 == j1 + 1) and i2 == i1

# Read input
Sx, Sy = map(int, input().split())
Tx, Ty = map(int, input().split())

i1 = Sx
j1 = Sy
i2 = Tx
j2 = Ty

if same_tile(i1, j1, i2, j2):
    print(0)
else:
    dx = abs(i1 - i2)
    dy = abs(j1 - j2)
    if (i1 + j1) % 2 == (i2 + j2) % 2:
        minimal = max(dx, dy)
    else:
        minimal = dx + dy
    print(minimal)