# Read the input
Sx, Sy = map(int, input().split())
Tx, Ty = map(int, input().split())

sum_par = Sx + Sy

same_tile = False

if sum_par % 2 == 0:
    # Check if Tx is between Sx and Sx + 1 (inclusive) and Ty == Sy
    if Sx <= Tx <= Sx + 1 and Ty == Sy:
        same_tile = True
else:
    # Check if Tx == Sx and Ty is between Sy and Sy + 1 (inclusive)
    if Tx == Sx and Sy <= Ty <= Sy + 1:
        same_tile = True

if same_tile:
    print(0)
else:
    dx = abs(Tx - Sx)
    dy = abs(Ty - Sy)
    print(max(dx, dy))