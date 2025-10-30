v1, v2, v3 = map(int, input().split())

if v1 + 2 * v2 + 3 * v3 != 1029:
    print("No")
    exit()

def check_and_generate(dx, dy, dz, swap_axis):
    a1, b1, c1 = 0, 0, 0
    if swap_axis == 0:
        a2, b2, c2 = 0, 7 - dy, 0
        a3, b3, c3 = 7 - dx, 0, 0
    elif swap_axis == 1:
        a2, b2, c2 = 0, 0, 7 - dz
        a3, b3, c3 = 7 - dx, 0, 0
    elif swap_axis == 2:
        a2, b2, c2 = 0, 7 - dy, 0
        a3, b3, c3 = 0, 0, 7 - dz
    else:
        return False, None
    coords = [a1, b1, c1, a2, b2, c2, a3, b3, c3]
    if all(abs(x) <= 100 for x in coords):
        return True, coords
    return False, None

# Try approach with z-axis aligned (x and y)
if v3 % 7 == 0:
    k = v3 // 7
    sum_xy = (v2 + 3 * v3) // 7
    if (v2 + 3 * v3) % 7 == 0:
        for dx in range(1, 8):
            if k % dx != 0:
                continue
            dy = k // dx
            if dy < 1 or dy > 7:
                continue
            if 7 * (dx + dy) + k == sum_xy:
                valid, coords = check_and_generate(dx, dy, 7, 0)
                if valid:
                    print("Yes")
                    print(" ".join(map(str, coords)))
                    exit()

# Try approach with y-axis aligned (x and z)
if v3 % 7 == 0:
    k = v3 // 7
    sum_xz = (v2 + 3 * v3) // 7
    if (v2 + 3 * v3) % 7 == 0:
        for dx in range(1, 8):
            if k % dx != 0:
                continue
            dz = k // dx
            if dz < 1 or dz > 7:
                continue
            if 7 * (dx + dz) + k == sum_xz:
                valid, coords = check_and_generate(dx, dz, 7, 1)
                if valid:
                    print("Yes")
                    print(" ".join(map(str, coords)))
                    exit()

# Try approach with x-axis aligned (y and z)
if v3 % 7 == 0:
    k = v3 // 7
    sum_yz = (v2 + 3 * v3) // 7
    if (v2 + 3 * v3) % 7 == 0:
        for dy in range(1, 8):
            if k % dy != 0:
                continue
            dz = k // dy
            if dz < 1 or dz > 7:
                continue
            if 7 * (dy + dz) + k == sum_yz:
                valid, coords = check_and_generate(dy, dz, 7, 2)
                if valid:
                    print("Yes")
                    print(" ".join(map(str, coords)))
                    exit()

print("No")