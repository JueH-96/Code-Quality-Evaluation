import sys
data = sys.stdin.read().split()
index = 0
Sx = int(data[index])
index += 1
Sy = int(data[index])
index += 1
Tx = int(data[index])
index += 1
Ty = int(data[index])

dx = abs(Tx - Sx)
dy = abs(Ty - Sy)

if dy > 0:
    toll = dy
else:  # dy == 0
    if dx == 0:
        toll = 0
    else:
        min_st = min(Sx, Tx)
        parity_first_m = (min_st + 1) % 2
        P_sy = Sy % 2
        if dx % 2 == 0:
            toll = dx // 2
        else:
            if parity_first_m == P_sy:
                toll = (dx + 1) // 2
            else:
                toll = dx // 2

print(toll)