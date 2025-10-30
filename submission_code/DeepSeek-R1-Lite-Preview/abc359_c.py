Sx, Sy = map(int, input().split())
Tx, Ty = map(int, input().split())

delta_x = abs(Tx - Sx)
delta_y = abs(Ty - Sy)

toll = (delta_x + 1) // 2 + (delta_y + 1) // 2

if delta_x < 2 and delta_y < 2:
    toll -= 1

print(toll)