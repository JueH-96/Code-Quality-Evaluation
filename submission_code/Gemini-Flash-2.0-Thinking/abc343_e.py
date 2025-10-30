def solve():
    v1, v2, v3 = map(int, input().split())
    s = 7
    cube_vol = s**3

    def intersection_volume(c1_start, c2_start):
        x_overlap = max(0, min(c1_start[0] + s, c2_start[0] + s) - max(c1_start[0], c2_start[0]))
        y_overlap = max(0, min(c1_start[1] + s, c2_start[1] + s) - max(c1_start[1], c2_start[1]))
        z_overlap = max(0, min(c1_start[2] + s, c2_start[2] + s) - max(c1_start[2], c2_start[2]))
        return x_overlap * y_overlap * z_overlap

    for a2 in range(-s + 1, s):
        for b2 in range(-s + 1, s):
            for c2 in range(-s + 1, s):
                for a3 in range(-s + 1, s):
                    for b3 in range(-s + 1, s):
                        for c3 in range(-s + 1, s):
                            c1_start = (0, 0, 0)
                            c2_start = (a2, b2, c2)
                            c3_start = (a3, b3, c3)

                            i12 = intersection_volume(c1_start, c2_start)
                            i13 = intersection_volume(c1_start, c3_start)
                            i23 = intersection_volume(c2_start, c3_start)

                            x_min = max(c1_start[0], c2_start[0], c3_start[0])
                            y_min = max(c1_start[1], c2_start[1], c3_start[1])
                            z_min = max(c1_start[2], c2_start[2], c3_start[2])
                            x_max = min(c1_start[0] + s, c2_start[0] + s, c3_start[0] + s)
                            y_max = min(c1_start[1] + s, c2_start[1] + s, c3_start[1] + s)
                            z_max = min(c1_start[2] + s, c2_start[2] + s, c3_start[2] + s)
                            i123 = max(0, x_max - x_min) * max(0, y_max - y_min) * max(0, z_max - z_min)

                            v3_calc = i123
                            v2_calc = (i12 - v3_calc) + (i13 - v3_calc) + (i23 - v3_calc)
                            v1_calc = cube_vol * 3 - 2 * (i12 + i13 + i23) + 3 * v3_calc

                            if round(v1_calc) == v1 and round(v2_calc) == v2 and round(v3_calc) == v3:
                                print("Yes")
                                print(f"{c1_start[0]} {c1_start[1]} {c1_start[2]} {c2_start[0]} {c2_start[1]} {c2_start[2]} {c3_start[0]} {c3_start[1]} {c3_start[2]}")
                                return

    print("No")

solve()