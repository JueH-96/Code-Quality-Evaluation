v1, v2, v3 = map(int, input().split())
side = 7
if v3 > side**3 or v2 > 3 * side**3 or v1 > 3 * side**3:
    print('No')
else:
    x = 0
    y = 0
    z = 0
    if v3 > 0:
        for i in range(1, side + 1):
            if v3 % i == 0:
                for j in range(1, side + 1):
                    if (v3 // i) % j == 0:
                        k = v3 // i // j
                        if k <= side:
                            x = i
                            y = j
                            z = k
                            break
                if x > 0:
                    break
    x2 = 0
    y2 = 0
    z2 = 0
    if v2 > 0:
        rem = v2
        if x > 0:
            for i in range(1, side + 1):
                if rem % (y * z) == 0:
                    j = rem // (y * z)
                    if j <= side - x:
                        x2 = j
                        rem -= j * y * z
                        break
        if y > 0 and rem > 0:
            for i in range(1, side + 1):
                if rem % (x * z) == 0:
                    j = rem // (x * z)
                    if j <= side - y:
                        y2 = j
                        rem -= j * x * z
                        break
        if z > 0 and rem > 0:
            for i in range(1, side + 1):
                if rem % (x * y) == 0:
                    j = rem // (x * y)
                    if j <= side - z:
                        z2 = j
                        rem -= j * x * y
                        break
        if rem > 0:
            print('No')
            exit()
    x1 = 0
    y1 = 0
    z1 = 0
    if v1 > 0:
        rem = v1
        for i in range(1, side + 1):
            if rem % (side * side) == 0:
                j = rem // (side * side)
                if j <= side - max(x + x2, y2):
                    x1 = j
                    rem -= j * side * side
                    break
        if rem > 0:
            for i in range(1, side + 1):
                if rem % (side * side) == 0:
                    j = rem // (side * side)
                    if j <= side - max(y + y2, x2):
                        y1 = j
                        rem -= j * side * side
                        break
        if rem > 0:
            for i in range(1, side + 1):
                if rem % (side * side) == 0:
                    j = rem // (side * side)
                    if j <= side - max(z + z2, x1, y1):
                        z1 = j
                        rem -= j * side * side
                        break
        if rem > 0:
            print('No')
            exit()
    print('Yes')
    print(0, 0, 0, x + x2, y, 0, x, 0, z + z2)