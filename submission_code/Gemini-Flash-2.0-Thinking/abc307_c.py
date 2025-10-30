def solve():
    h_a, w_a = map(int, input().split())
    a = [input() for _ in range(h_a)]
    h_b, w_b = map(int, input().split())
    b = [input() for _ in range(h_b)]
    h_x, w_x = map(int, input().split())
    x = [input() for _ in range(h_x)]

    black_a = []
    for r in range(h_a):
        for c in range(w_a):
            if a[r][c] == '#':
                black_a.append((r, c))

    black_b = []
    for r in range(h_b):
        for c in range(w_b):
            if b[r][c] == '#':
                black_b.append((r, c))

    for r_a_x in range(-(h_a - 1), h_x):
        for c_a_x in range(-(w_a - 1), w_x):
            for r_b_x in range(-(h_b - 1), h_x):
                for c_b_x in range(-(w_b - 1), w_x):
                    combined = [['.' for _ in range(w_x)] for _ in range(h_x)]

                    # Paste A
                    for r_a, c_a in black_a:
                        row = r_a + r_a_x
                        col = c_a + c_a_x
                        if 0 <= row < h_x and 0 <= col < w_x:
                            combined[row][col] = '#'

                    # Paste B
                    for r_b, c_b in black_b:
                        row = r_b + r_b_x
                        col = c_b + c_b_x
                        if 0 <= row < h_x and 0 <= col < w_x:
                            combined[row][col] = '#'

                    combined_strs = ["".join(row) for row in combined]

                    if combined_strs == x:
                        includes_all_black = True
                        for r_a, c_a in black_a:
                            row = r_a + r_a_x
                            col = c_a + c_a_x
                            if not (0 <= row < h_x and 0 <= col < w_x and x[row][col] == '#'):
                                includes_all_black = False
                                break
                        if not includes_all_black:
                            continue

                        for r_b, c_b in black_b:
                            row = r_b + r_b_x
                            col = c_b + c_b_x
                            if not (0 <= row < h_x and 0 <= col < w_x and x[row][col] == '#'):
                                includes_all_black = False
                                break
                        if not includes_all_black:
                            continue

                        print("Yes")
                        return

    print("No")

solve()