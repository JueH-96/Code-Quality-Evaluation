V1, V2, V3 = map(int, input().split())

if V1 + V2 + V3 > 3 * 7 ** 3 or V1 % 7 != 0 or V2 % 7 != 0 or V3 % 7 != 0:
    print('No')
else:
    # All conditions can be satisfied via shifting of each cube in one direction by 7.
    a1, b1, c1 = 0, 0, 0
    a2, b2, c2 = 0, 7, 0
    a3, b3, c3 = 7, 0, 0
    
    V1_check = V1 == 3 * 7 ** 3 - (2 * 7 ** 2 + 7) - (2 * 7 ** 2) - V2 - V3
    V2_check = V2 == (2 * 7 ** 2 + 7)
    V3_check = V3 == 7
    
    # Check if this configuration satisfies the volumes
    if V1_check and V2_check and V3_check:
        print('Yes')
        print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
    else:
        print('No')