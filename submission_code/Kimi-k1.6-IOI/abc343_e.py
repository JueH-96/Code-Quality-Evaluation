V1, V2, V3 = map(int, input().split())

total = V1 + 2 * V2 + 3 * V3
if total != 3 * 7 * 7 * 7:
    print("No")
else:
    if V3 % 7 != 0 or V2 % 7 != 0:
        print("No")
    else:
        A3 = V3 // 7
        v2_7 = V2 // 7
        found = False
        dx_final = None
        dy_final = None
        
        if A3 == 0:
            if v2_7 % 7 == 0:
                d = v2_7 // 7
                if 0 <= d <= 7:
                    dx_final = d
                    dy_final = 0
                    found = True
        else:
            for dx in range(1, 8):  # dx from 1 to 7
                if A3 % dx != 0:
                    continue
                dy = A3 // dx
                if dy < 0 or dy > 7:
                    continue
                if 7 * (dx + dy) == v2_7 + 2 * A3:
                    dx_final = dx
                    dy_final = dy
                    found = True
                    break
        
        if not found:
            print("No")
        else:
            print("Yes")
            a1, b1, c1 = 0, 0, 0
            a2, b2, c2 = 0, 7 - dy_final, 0
            a3, b3, c3 = 7 - dx_final, 0, 0
            print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")