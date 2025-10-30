V1, V2, V3 = map(int, input().split())

if V1 + 2 * V2 + 3 * V3 != 1029:
    print("No")
else:
    if V3 % 7 != 0:
        print("No")
    else:
        T = V3
        k = T // 7
        found = False
        
        if T == 0:
            if V2 == 0 and V1 == 1029:
                a1, b1, c1 = 0, 0, 0
                a2, b2, c2 = 7, 0, 0
                a3, b3, c3 = 14, 0, 0
                if all(abs(x) <= 100 for x in [a1, b1, c1, a2, b2, c2, a3, b3, c3]):
                    print("Yes")
                    print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
                    found = True
        else:
            for a in range(1, k + 1):
                if k % a != 0:
                    continue
                b = k // a
                s = 7 - b
                t = 7 - a
                sum_pair = 49 * ((7 - s) + (7 - t)) + 7
                computed_V2 = sum_pair - 3 * T
                computed_V1 = 1029 - 2 * computed_V2 - 3 * T
                if computed_V1 == V1 and computed_V2 == V2:
                    if abs(s) <= 100 and abs(t) <= 100:
                        print("Yes")
                        print(f"0 0 0 0 {s} 0 {t} 0 0")
                        found = True
                        break
        
        if not found:
            print("No")