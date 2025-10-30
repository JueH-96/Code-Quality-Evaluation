Sx, Sy = map(int, input().split())
Tx, Ty = map(int, input().split())

# Check if same tile
same = False
if Sx == Tx and Sy == Ty:
    same = True
elif (Sx + Sy) % 2 == 0:
    if Tx == Sx + 1 and Ty == Sy:
        same = True
else:
    if Tx == Sx and Ty == Sy + 1:
        same = True

if same:
    print(0)
else:
    dx = abs(Sx - Tx)
    dy = abs(Sy - Ty)
    print(max(dx, dy))