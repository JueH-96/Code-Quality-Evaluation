V1, V2, V3 = map(int, input().split())

# Check if the initial condition is satisfied
if V1 != 1029 - 2 * V2 - 3 * V3:
    print("No")
else:
    if V3 == 0:
        # Handle V3=0 case
        if V2 % 7 != 0:
            print("No")
        else:
            k = V2 // 7
            target = k + 49
            # Find factors >=8 for target
            factors = []
            for i in range(8, int(target**0.5) + 1):
                if target % i == 0:
                    j = target // i
                    if j >= 8:
                        factors.append((i, j))
            found = False
            a = -1
            b = -1
            for f1, f2 in factors:
                a_candidate = f1 - 7
                b_candidate = f2 - 7
                if 1 <= a_candidate <= 7 and 1 <= b_candidate <= 7:
                    a = a_candidate
                    b = b_candidate
                    found = True
                    break
                # Try swapping factors
                a_candidate = f2 - 7
                b_candidate = f1 - 7
                if 1 <= a_candidate <= 7 and 1 <= b_candidate <= 7:
                    a = a_candidate
                    b = b_candidate
                    found = True
                    break
            if found:
                # Check triple overlap
                # Cube1: (0,0,0), Cube2: (7-a,0,0), Cube3: (0,7-b,0)
                # Calculate triple overlap
                a1, b1, c1 = 0, 0, 0
                a2, b2, c2 = 7 - a, 0, 0
                a3, b3, c3 = 0, 7 - b, 0
                # Calculate overlaps
                x_overlap_min = max(a1, a2, a3)
                x_overlap_max = min(a1 + 7, a2 + 7, a3 + 7)
                x_overlap = max(0, x_overlap_max - x_overlap_min)
                y_overlap_min = max(b1, b2, b3)
                y_overlap_max = min(b1 + 7, b2 + 7, b3 + 7)
                y_overlap = max(0, y_overlap_max - y_overlap_min)
                z_overlap_min = max(c1, c2, c3)
                z_overlap_max = min(c1 + 7, c2 + 7, c3 + 7)
                z_overlap = max(0, z_overlap_max - z_overlap_min)
                triple_overlap = x_overlap * y_overlap * z_overlap
                if triple_overlap != 0:
                    found = False
            if found:
                a1, b1, c1 = 0, 0, 0
                a2, b2, c2 = 7 - a, 0, 0
                a3, b3, c3 = 0, 7 - b, 0
                if all(abs(x) <= 100 for x in [a1, b1, c1, a2, b2, c2, a3, b3, c3]):
                    print("Yes")
                    print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
                else:
                    print("No")
            else:
                print("No")
        pass
    else:
        # Proceed when V3 != 0
        if V3 % 7 != 0:
            print("No")
        else:
            required_sum = V2 + 2 * V3
            if required_sum % 49 != 0:
                print("No")
            else:
                s = required_sum // 49
                product = V3 // 7
                # Find dx and dy such that dx + dy = s and dx * dy = product
                found = False
                dx = -1
                dy = -1
                for dx_candidate in range(1, 8):
                    for dy_candidate in range(1, 8):
                        if dx_candidate + dy_candidate == s and dx_candidate * dy_candidate == product:
                            dx = dx_candidate
                            dy = dy_candidate
                            found = True
                            break
                    if found:
                        break
                if found:
                    a1, b1, c1 = 0, 0, 0
                    a2, b2, c2 = 0, 7 - dy, 0
                    a3, b3, c3 = 7 - dx, 0, 0
                    # Check if all coordinates are within the limits
                    if all(abs(x) <= 100 for x in [a1, b1, c1, a2, b2, c2, a3, b3, c3]):
                        print("Yes")
                        print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
                    else:
                        print("No")
                else:
                    print("No")