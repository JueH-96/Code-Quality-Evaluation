import math

v1, v2, v3 = map(int, input().split())

# Check the inclusion-exclusion condition
if v1 != 1029 - 2 * v2 - 3 * v3:
    print("No")
else:
    if v3 % 7 != 0:
        print("No")
    else:
        p = v3 // 7
        numerator = v2 + 2 * v3
        if numerator % 49 != 0:
            print("No")
        else:
            s = numerator // 49
            if s < 0 or s > 14:
                print("No")
            else:
                discriminant = s * s - 4 * p
                if discriminant < 0:
                    print("No")
                else:
                    sqrt_discriminant = int(math.isqrt(discriminant))
                    if sqrt_discriminant * sqrt_discriminant != discriminant:
                        print("No")
                    else:
                        found = False
                        solutions = []
                        # Check both possibilities for dx: (s + sqrt_d)/2 and (s - sqrt_d)/2
                        for sign in [1, -1]:
                            numerator_sol = s + sign * sqrt_discriminant
                            if numerator_sol % 2 == 0:
                                dx = numerator_sol // 2
                                dy = s - dx
                                if 0 <= dx <= 7 and 0 <= dy <= 7:
                                    solutions.append((dx, dy))
                        if not solutions:
                            print("No")
                        else:
                            dx, dy = solutions[0]
                            # Generate cube coordinates
                            a1, b1, c1 = 0, 0, 0
                            a2, b2, c2 = 0, 7 - dy, 0
                            a3, b3, c3 = 7 - dx, 0, 0
                            print("Yes")
                            print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")