ha, wa = map(int, input().split())
a = [input().strip() for _ in range(ha)]
hb, wb = map(int, input().split())
b = [input().strip() for _ in range(hb)]
hx, wx = map(int, input().split())
x = [input().strip() for _ in range(hx)]

black_a = [(i, j) for i in range(ha) for j in range(wa) if a[i][j] == '#']
black_b = [(i, j) for i in range(hb) for j in range(wb) if b[i][j] == '#']
black_x = {(i, j) for i in range(hx) for j in range(wx) if x[i][j] == '#'}

possible_a = True
if black_a:
    lower_dx_a = max(-i for i, j in black_a)
    upper_dx_a = min((hx - 1) - i for i, j in black_a)
    lower_dy_a = max(-j for i, j in black_a)
    upper_dy_a = min((wx - 1) - j for i, j in black_a)
    possible_a = (lower_dx_a <= upper_dx_a) and (lower_dy_a <= upper_dy_a)
else:
    possible_a = False

possible_b = True
if black_b:
    lower_dx_b = max(-i for i, j in black_b)
    upper_dx_b = min((hx - 1) - i for i, j in black_b)
    lower_dy_b = max(-j for i, j in black_b)
    upper_dy_b = min((wx - 1) - j for i, j in black_b)
    possible_b = (lower_dx_b <= upper_dx_b) and (lower_dy_b <= upper_dy_b)
else:
    possible_b = False

if not possible_a or not possible_b:
    print("No")
else:
    found = False
    for dx_a in range(lower_dx_a, upper_dx_a + 1):
        for dy_a in range(lower_dy_a, upper_dy_a + 1):
            a_cover = set((dx_a + i, dy_a + j) for i, j in black_a)
            for dx_b in range(lower_dx_b, upper_dx_b + 1):
                for dy_b in range(lower_dy_b, upper_dy_b + 1):
                    b_cover = set((dx_b + i, dy_b + j) for i, j in black_b)
                    combined = a_cover.union(b_cover)
                    if combined == black_x:
                        found = True
                        print("Yes")
                        break  # Exit all loops
                if found:
                    break
            if found:
                break
        if found:
            break
    if not found:
        print("No")