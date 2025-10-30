Sx, Sy = map(int, input().split())
Tx, Ty = map(int, input().split())

# Determine the start tile's coverage
if (Sx + Sy) % 2 == 0:
    # Horizontal tile: covers x from Sx to Sx+2, y from Sy to Sy+1
    x_min = Sx
    x_max = Sx + 2
    y_min = Sy
    y_max = Sy + 1
else:
    # Vertical tile: covers x from Sx to Sx+1, y from Sy to Sy+2
    x_min = Sx
    x_max = Sx + 1
    y_min = Sy
    y_max = Sy + 2

# Check if the end point is within the start tile's coverage
x = Tx + 0.5
y = Ty + 0.5

if (x_min <= x <= x_max) and (y_min <= y <= y_max):
    print(0)
else:
    dx = abs(Sx - Tx)
    dy = abs(Sy - Ty)
    print(max(dx, dy))