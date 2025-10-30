# YOUR CODE HERE
V1_input, V2_input, V3_input = map(int, input().split())

V_cube = 343  # Volume of one cube
total_volume = 3 * V_cube  # Total volume of all cubes without overlap
required_sum = V1_input + 2 * V2_input + 3 * V3_input

if required_sum != total_volume:
    print("No")
    exit()

shifts = [-7, -6, -1, 0, 1, 6, 7]

found = False
for dx2 in shifts:
    for dy2 in shifts:
        for dz2 in shifts:
            a2, b2, c2 = dx2, dy2, dz2
            for dx3 in shifts:
                for dy3 in shifts:
                    for dz3 in shifts:
                        a3, b3, c3 = dx3, dy3, dz3

                        # Positions of the cubes
                        a1, b1, c1 = 0, 0, 0

                        # Compute overlaps along each axis
                        def overlap(a_start1, a_end1, a_start2, a_end2):
                            return max(0, min(a_end1, a_end2) - max(a_start1, a_start2))

                        # Overlaps between cube1 and cube2
                        lx12 = overlap(a1, a1+7, a2, a2+7)
                        ly12 = overlap(b1, b1+7, b2, b2+7)
                        lz12 = overlap(c1, c1+7, c2, c2+7)
                        V12 = lx12 * ly12 * lz12

                        # Overlaps between cube1 and cube3
                        lx13 = overlap(a1, a1+7, a3, a3+7)
                        ly13 = overlap(b1, b1+7, b3, b3+7)
                        lz13 = overlap(c1, c1+7, c3, c3+7)
                        V13 = lx13 * ly13 * lz13

                        # Overlaps between cube2 and cube3
                        lx23 = overlap(a2, a2+7, a3, a3+7)
                        ly23 = overlap(b2, b2+7, b3, b3+7)
                        lz23 = overlap(c2, c2+7, c3, c3+7)
                        V23 = lx23 * ly23 * lz23

                        # Overlap among all three cubes
                        lx123 = overlap(max(a1, a2, a3), min(a1+7, a2+7, a3+7), max(a1, a2, a3), min(a1+7, a2+7, a3+7))
                        ly123 = overlap(max(b1, b2, b3), min(b1+7, b2+7, b3+7), max(b1, b2, b3), min(b1+7, b2+7, b3+7))
                        lz123 = overlap(max(c1, c2, c3), min(c1+7, c2+7, c3+7), max(c1, c2, c3), min(c1+7, c2+7, c3+7))
                        V123 = lx123 * ly123 * lz123

                        # Calculate V2 and V3
                        V3 = V123
                        V2 = V12 + V13 + V23 - 3 * V123
                        V1 = 3 * V_cube - 2 * (V12 + V13 + V23) + 3 * V123

                        # Check if V1, V2, V3 match the input
                        if V1 == V1_input and V2 == V2_input and V3 == V3_input:
                            a_values = [a1, b1, c1, a2, b2, c2, a3, b3, c3]
                            if all(-100 <= x <= 100 for x in a_values):
                                print("Yes")
                                print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
                                found = True
                                exit()
if not found:
    print("No")